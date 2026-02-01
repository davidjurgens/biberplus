import os
import sys

import torch
import torch.nn as nn
from accelerate import Accelerator
from torch.optim import AdamW
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoModelForMaskedLM,
    get_cosine_schedule_with_warmup,
)

sys.path.append("../..")

from src.custom_training.model_utils import (
    encode_batches,
    load_model,
    setup_accelerator,
)
from src.custom_training.trainer_utils import get_dataloaders
from src.hiatus_training.losses import InfoNCE_loss_full


class ContrastiveModel(nn.Module):
    def __init__(self, args, wandb=None):
        super(ContrastiveModel, self).__init__()
        self.args = args
        self.device = args.device
        self.train_loader, self.eval_loader = get_dataloaders(
            args, mlm=False, pairs=True
        )
        self.wandb = wandb

    def init_model(self):
        self.accelerator = Accelerator(gradient_accumulation_steps=self.args.grad_acc)
        self.device = self.accelerator.device
        self.model = AutoModel.from_pretrained(self.args.pretrained_model).to(
            self.device
        )

        optimizer = AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.args.num_warmup_steps,
            num_training_steps=len(self.train_loader) * self.args.epochs,
            num_cycles=1,
        )

        self.model, self.optimizer, self.train_loader, self.scheduler = (
            self.accelerator.prepare(
                self.model, optimizer, self.train_loader, scheduler
            )
        )

    def train_model(self):
        args = self.args
        self.init_model()
        self.model.train()

        best_eval_metric = float("inf")
        running_loss = 0

        for epoch in range(args.epochs):
            for i, (query_batch, candidate_batch) in tqdm(
                enumerate(self.train_loader), total=len(self.train_loader)
            ):
                with self.accelerator.accumulate(self.model):
                    loss = self.train_step(query_batch, candidate_batch)
                    running_loss += loss.item()

                if i % args.grad_acc == 0 and i > 0:
                    train_log = {"Train Loss": running_loss / args.grad_acc}
                    if self.wandb:
                        self.wandb.log(train_log)
                    print(train_log)
                    running_loss = 0

                if i % (args.saving_step * args.grad_acc) == 0 and i > 0:
                    if args.evaluate:
                        eval_metric = self.evaluate()
                        if eval_metric < best_eval_metric:
                            best_eval_metric = eval_metric
                            self.save_model(step=str(i), version="best")
                        else:
                            self.save_model(step=str(i), version="last")

        self.accelerator.end_training()

    def train_step(self, query_batch, candidate_batch):
        z1 = self.model(**query_batch).pooler_output
        z2 = self.model(**candidate_batch).pooler_output
        loss = InfoNCE_loss_full(z1, z2) / self.args.grad_acc
        self.accelerator.backward(loss)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        return loss

    def evaluate(self):
        print("Evaluating the model...")
        # TODO: Implement HIATUS evaluation here...
        self.model.eval()
        losses = []
        for i, (query_batch, candidate_batch, query_authors, target_authors) in tqdm(
            enumerate(self.eval_loader),
            total=len(self.eval_loader),
            position=0,
            leave=True,
        ):
            with torch.no_grad():
                z1 = self.model(**query_batch.to(self.device)).pooler_output
                z2 = self.model(**candidate_batch.to(self.device)).pooler_output
                loss = InfoNCE_loss_full(z1, z2) / self.args.grad_acc
                losses.append(
                    self.accelerator.gather(loss.repeat(self.args.batch_size))
                )

        losses = torch.cat(losses)[: len(self.eval_loader)]

        eval_log = {"Eval Loss": torch.mean(losses).item()}
        print(eval_log)

        if self.wandb:
            self.wandb.log(eval_log)

        return torch.mean(losses)

    def save_model(self, step, version="last"):
        """Save model checkpoint"""
        checkpoint_dir = os.path.join(
            self.args.out_dir, f"{self.args.run_name}", f"{version}"
        )
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save the LM
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(checkpoint_dir)

        # Save the optimizer and scheduler
        torch.save(
            {
                "step": step,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
            },
            os.path.join(checkpoint_dir, "optimizer_and_scheduler.pt"),
        )

        print(f"Saved {version} model to {checkpoint_dir}")


class StyleContrastiveModel(nn.Module):
    def __init__(
        self, args, biber_plus_size=192, adversarial_loss_weight=0.1, wandb=None
    ):
        super(StyleContrastiveModel, self).__init__()
        self.args = args
        self.style_dimensions = args.style_dimensions
        self.adversarial_loss_weight = adversarial_loss_weight
        self.biber_plus_size = biber_plus_size
        self.device = args.device
        self.train_loader, self.eval_loader = get_dataloaders(
            args, mlm=True, pairs=False
        )
        self.wandb = wandb

    def init_model(self):
        """Initialize model in the training loop to accommodate accelerate and the multi-GPU setup"""
        self.accelerator = Accelerator(gradient_accumulation_steps=self.args.grad_acc)
        self.device = self.accelerator.device
        if self.args.resume:
            checkpoint_dir = os.path.join(
                self.args.out_dir, f"{self.args.run_name}", "last"
            )
            self.model, optimizer, scheduler = load_model(checkpoint_dir)

        else:
            self.model = AutoModelForMaskedLM.from_pretrained(
                self.args.pretrained_model
            ).to(self.device)

            optimizer = AdamW(
                self.model.parameters(),
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
            )
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.args.num_warmup_steps,
                num_training_steps=len(self.train_loader) * self.args.epochs,
                num_cycles=1,
            )

        self.style_ll = nn.Linear(
            in_features=self.style_dimensions,
            out_features=self.biber_plus_size,
            device=self.device,
        )
        self.not_style_ll = nn.Linear(
            in_features=self.model.config.hidden_size - self.style_dimensions,
            out_features=self.biber_plus_size,
            device=self.device,
        )
        self.cls_dropout = nn.Dropout(p=0.2)
        self.model, self.optimizer, self.train_loader, self.scheduler = (
            self.accelerator.prepare(
                self.model, optimizer, self.train_loader, scheduler
            )
        )

    def train_model(self):
        args = self.args
        self.init_model()
        self.model.train()

        (
            running_loss,
            running_contrastive_loss,
            running_style_loss,
            running_adversarial_loss,
        ) = 0, 0, 0, 0
        best_eval_metric = float("inf")

        for _ in range(args.epochs):
            for i, (batchA, batchB, biberEncodingsA, biberEncodingsB) in tqdm(
                enumerate(self.train_loader), total=len(self.train_loader)
            ):
                with self.accelerator.accumulate(self.model):
                    loss, contrastive_loss, style_loss, adversarial_loss = (
                        self.train_step(
                            batchA, batchB, biberEncodingsA, biberEncodingsB
                        )
                    )
                    running_loss += loss.item()
                    running_contrastive_loss += contrastive_loss.item()
                    running_style_loss += style_loss.item()
                    running_adversarial_loss += adversarial_loss.item()

                    if i % args.grad_acc == 0 and i > 0:
                        if self.wandb:
                            train_log = {
                                "Train Overall Loss": running_loss / args.grad_acc,
                                "Train Contrastive Loss": running_contrastive_loss
                                / args.grad_acc,
                                "Train Style Loss": running_style_loss / args.grad_acc,
                                "Train Adversarial Training Loss": running_adversarial_loss
                                / args.grad_acc,
                            }
                            self.wandb.log(train_log)
                            print(train_log)
                        (
                            running_loss,
                            running_contrastive_loss,
                            running_style_loss,
                            running_adversarial_loss,
                        ) = 0, 0, 0, 0

                    if i % (args.saving_step * args.grad_acc) == 0 and i > 0:
                        if args.evaluate:
                            eval_style_loss = self.evaluate()
                            if eval_style_loss < best_eval_metric:
                                best_eval_metric = eval_style_loss
                                self.save_model(step=str(i), version="best")
                            else:
                                self.save_model(step=str(i), version="last")

            self.accelerator.end_training()

    def train_step(self, batchA, batchB, biberEncodingsA, biberEncodingsB):
        outputsA = self.model(**batchA.to(self.device), output_hidden_states=True)
        outputsB = self.model(**batchB.to(self.device), output_hidden_states=True)

        z1, z2 = outputsA.pooler_output, outputsB.pooler_output

        biberEncodingsA = torch.FloatTensor(biberEncodingsA).to(self.device)
        biberEncodingsB = torch.FloatTensor(biberEncodingsB).to(self.device)
        biberLossA, biberLossB = (
            self.biber_style_loss(z1, biberEncodingsA),
            self.biber_style_loss(z2, biberEncodingsB),
        )
        adversarial_lossA, adversarial_lossB = (
            self.biber_adversarial_loss(z1, biberEncodingsA),
            self.biber_adversarial_loss(z2, biberEncodingsB),
        )

        contrastive_loss = self.contrastive_loss(z1, z2)
        style_loss = biberLossA + biberLossB
        adversarial_loss = adversarial_lossA + adversarial_lossB

        loss = contrastive_loss + style_loss + adversarial_loss

        self.accelerator.backward(loss)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        return loss, contrastive_loss, style_loss, adversarial_loss

    def contrastive_loss(self, z1, z2):
        return (1 - self.adversarial_loss_weight) * InfoNCE_loss_full(z1, z2)

    def biber_style_loss(self, z, biber_encoding):
        style_dimensions = z[:, : self.style_dimensions]
        style_output = self.style_ll(style_dimensions)
        return nn.MSELoss()(style_output, biber_encoding)

    def biber_adversarial_loss(self, z, biber_encoding):
        not_style_dimensions = z[:, self.style_dimensions :]
        not_style_output = self.not_style_ll(not_style_dimensions)
        return (
            -1 * nn.MSELoss()(not_style_output, biber_encoding)
        ) * self.adversarial_loss_weight
