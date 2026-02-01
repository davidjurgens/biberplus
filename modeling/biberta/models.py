import os
import sys

import torch
import torch.nn as nn
from accelerate import Accelerator
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, get_cosine_schedule_with_warmup

sys.path.append("../..")

from src.custom_training.model_utils import calculate_perplexity, load_model
from src.custom_training.trainer_utils import get_dataloaders


class StyleMLM(nn.Module):
    def __init__(
        self, args, biber_plus_size=192, adversarial_loss_weight=0.1, wandb=None
    ):
        super(StyleMLM, self).__init__()
        self.args = args
        self.style_dimensions = args.style_dimensions
        self.adversarial_loss_weight = adversarial_loss_weight
        self.biber_plus_size = biber_plus_size
        self.device = args.device
        self.train_loader, self.eval_loader = get_dataloaders(
            args, mlm=True, pairs=False, biber=True
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

        running_loss, running_mlm_loss, running_style_loss, running_adversarial_loss = (
            0,
            0,
            0,
            0,
        )
        best_eval_metric = float("inf")

        for epoch in range(args.epochs):
            for i, (batch, biber_encoding) in tqdm(
                enumerate(self.train_loader), total=len(self.train_loader)
            ):
                with self.accelerator.accumulate(self.model):
                    loss, mlm_loss, style_loss, adversarial_loss = self.train_step(
                        batch, biber_encoding
                    )
                    running_loss += loss.item()
                    running_mlm_loss += mlm_loss.item()
                    running_style_loss += style_loss.item()
                    running_adversarial_loss += adversarial_loss.item()

                if i % args.grad_acc == 0 and i > 0:
                    if self.wandb:
                        train_log = {
                            "Train Overall Loss": running_loss / args.grad_acc,
                            "Train MLM Loss": running_mlm_loss / args.grad_acc,
                            "Train Style Loss": running_style_loss / args.grad_acc,
                            "Train Adversarial Training Loss": running_adversarial_loss
                            / args.grad_acc,
                        }
                        self.wandb.log(train_log)
                        print(train_log)
                    (
                        running_loss,
                        running_mlm_loss,
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

    def train_step(self, batch, biber_encoding):
        outputs = self.model(**batch, output_hidden_states=True)
        cls = self._get_cls(outputs, training=True)

        mlm_loss = (1 - self.adversarial_loss_weight) * outputs.loss
        style_loss = self.biber_style_loss(cls, biber_encoding)
        adversarial_loss = self.biber_adversarial_loss(cls, biber_encoding)

        loss = mlm_loss + style_loss + adversarial_loss

        self.accelerator.backward(loss)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        return loss, mlm_loss, style_loss, adversarial_loss

    def biber_style_loss(self, cls, biber_encoding):
        biber_encoding = torch.FloatTensor(biber_encoding).to(self.device)
        style_dimensions = cls[:, : self.style_dimensions]
        style_output = self.style_ll(style_dimensions)
        return nn.MSELoss()(style_output, biber_encoding)

    def biber_adversarial_loss(self, cls, biber_encoding):
        biber_encoding = torch.FloatTensor(biber_encoding).to(self.device)
        not_style_dimensions = cls[:, self.style_dimensions :]
        not_style_output = self.not_style_ll(not_style_dimensions)
        adversarial_loss = (
            -1 * nn.MSELoss()(not_style_output, biber_encoding)
        ) * self.adversarial_loss_weight
        return adversarial_loss

    def evaluate(self):
        print("Evaluating the model...")
        self.model.eval()
        mlm_losses, style_losses, adversarial_losses = [], [], []
        for batch, biber_encoding in tqdm(
            self.eval_loader, total=len(self.eval_loader), position=0, leave=True
        ):
            with torch.no_grad():
                outputs = self.model(**batch.to(self.device))
                mlm_loss = outputs.loss
                style_loss, adversarial_loss = self.get_biber_losses(
                    batch, biber_encoding
                )
                mlm_losses.append(
                    self.accelerator.gather(mlm_loss.repeat(self.args.batch_size))
                )
                style_losses.append(
                    self.accelerator.gather(style_loss.repeat(self.args.batch_size))
                )
                adversarial_losses.append(
                    self.accelerator.gather(
                        adversarial_loss.repeat(self.args.batch_size)
                    )
                )

        mlm_losses = torch.cat(mlm_losses)[: len(self.eval_loader)]
        style_losses = torch.cat(style_losses)[: len(self.eval_loader)]
        adversarial_losses = torch.cat(adversarial_losses)[: len(self.eval_loader)]
        perplexity = calculate_perplexity(mlm_losses)

        eval_log = {
            "Eval Perplexity": perplexity,
            "Eval MLM Loss": torch.mean(mlm_losses).item(),
            "Eval Style Loss": torch.mean(style_losses).item(),
            "Eval Adversarial Loss": torch.mean(adversarial_losses).item(),
        }
        print(eval_log)

        if self.wandb:
            self.wandb.log(eval_log)

        return torch.mean(style_losses)

    def get_biber_losses(self, batch, biber_encoding):
        outputs = self.model(**batch.to(self.device), output_hidden_states=True)
        cls = self._get_cls(outputs, training=False)
        style_loss = self.biber_style_loss(cls, biber_encoding)
        adversarial_loss = self.biber_adversarial_loss(cls, biber_encoding)
        return style_loss, adversarial_loss

    def _get_cls(self, outputs, training=True):
        last_hidden_state = outputs["hidden_states"][-1]
        cls_representation = last_hidden_state[:, 0, :]
        if training:
            return self.cls_dropout(cls_representation)
        return cls_representation

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


class StyleMLMPairs(nn.Module):
    def __init__(
        self, args, biber_plus_size=192, adversarial_loss_weight=0.1, wandb=None
    ):
        super(StyleMLMPairs, self).__init__()
        self.args = args
        self.style_dimensions = args.style_dimensions
        self.adversarial_loss_weight = adversarial_loss_weight
        self.biber_plus_size = biber_plus_size
        self.device = args.device
        self.train_loader, self.eval_loader = get_dataloaders(
            args, mlm=True, pairs=True
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

        running_loss, running_mlm_loss, running_style_loss, running_adversarial_loss = (
            0,
            0,
            0,
            0,
        )
        best_eval_metric = float("inf")
        index = 0

        for epoch in range(args.epochs):
            for query_batch, candidate_batch, query_biber, candidate_biber in tqdm(
                self.train_loader, total=len(self.train_loader)
            ):
                for batch, biber_encoding in [
                    (query_batch, query_biber),
                    (candidate_batch, candidate_biber),
                ]:
                    with self.accelerator.accumulate(self.model):
                        loss, mlm_loss, style_loss, adversarial_loss = self.train_step(
                            batch, biber_encoding
                        )
                        running_loss += loss.item()
                        running_mlm_loss += mlm_loss.item()
                        running_style_loss += style_loss.item()
                        running_adversarial_loss += adversarial_loss.item()
                        index += 1

                    if index % args.grad_acc == 0 and index > 0:
                        if self.wandb:
                            train_log = {
                                "Train Overall Loss": running_loss / args.grad_acc,
                                "Train MLM Loss": running_mlm_loss / args.grad_acc,
                                "Train Style Loss": running_style_loss / args.grad_acc,
                                "Train Adversarial Training Loss": running_adversarial_loss
                                / args.grad_acc,
                            }
                            self.wandb.log(train_log)
                            print(train_log)
                        (
                            running_loss,
                            running_mlm_loss,
                            running_style_loss,
                            running_adversarial_loss,
                        ) = 0, 0, 0, 0

                    if index % (args.saving_step * args.grad_acc) == 0 and index > 0:
                        if args.evaluate:
                            eval_style_loss = self.evaluate()
                            if eval_style_loss < best_eval_metric:
                                best_eval_metric = eval_style_loss
                                self.save_model(step=str(index), version="best")
                            else:
                                self.save_model(step=str(index), version="last")

        self.accelerator.end_training()

    def train_step(self, batch, biber_encoding):
        outputs = self.model(**batch, output_hidden_states=True)
        cls = self._get_cls(outputs, training=True)

        mlm_loss = (1 - self.adversarial_loss_weight) * outputs.loss
        style_loss = self.biber_style_loss(cls, biber_encoding)
        adversarial_loss = self.biber_adversarial_loss(cls, biber_encoding)

        loss = mlm_loss + style_loss + adversarial_loss

        self.accelerator.backward(loss)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        return loss, mlm_loss, style_loss, adversarial_loss

    def biber_style_loss(self, cls, biber_encoding):
        biber_encoding = torch.FloatTensor(biber_encoding).to(self.device)
        style_dimensions = cls[:, : self.style_dimensions]
        style_output = self.style_ll(style_dimensions)
        return nn.MSELoss()(style_output, biber_encoding)

    def biber_adversarial_loss(self, cls, biber_encoding):
        biber_encoding = torch.FloatTensor(biber_encoding).to(self.device)
        not_style_dimensions = cls[:, self.style_dimensions :]
        not_style_output = self.not_style_ll(not_style_dimensions)
        adversarial_loss = (
            -1 * nn.MSELoss()(not_style_output, biber_encoding)
        ) * self.adversarial_loss_weight
        return adversarial_loss

    def evaluate(self):
        print("Evaluating the model...")
        self.model.eval()
        mlm_losses, style_losses, adversarial_losses = [], [], []
        for (
            query_batch,
            candidate_batch,
            query_biber,
            candidate_biber,
            _,
            _,
        ) in tqdm(
            self.eval_loader, total=len(self.eval_loader), position=0, leave=True
        ):
            for batch, biber_encoding in [
                (query_batch, query_biber),
                (candidate_batch, candidate_biber),
            ]:
                with torch.no_grad():
                    outputs = self.model(**batch.to(self.device))
                    mlm_loss = outputs.loss
                    style_loss, adversarial_loss = self.get_biber_losses(
                        batch, biber_encoding
                    )
                    mlm_losses.append(
                        self.accelerator.gather(mlm_loss.repeat(self.args.batch_size))
                    )
                    style_losses.append(
                        self.accelerator.gather(style_loss.repeat(self.args.batch_size))
                    )
                    adversarial_losses.append(
                        self.accelerator.gather(
                            adversarial_loss.repeat(self.args.batch_size)
                        )
                    )

        mlm_losses = torch.cat(mlm_losses)[: len(self.eval_loader)]
        style_losses = torch.cat(style_losses)[: len(self.eval_loader)]
        adversarial_losses = torch.cat(adversarial_losses)[: len(self.eval_loader)]
        perplexity = calculate_perplexity(mlm_losses)

        eval_log = {
            "Eval Perplexity": perplexity,
            "Eval MLM Loss": torch.mean(mlm_losses).item(),
            "Eval Style Loss": torch.mean(style_losses).item(),
            "Eval Adversarial Loss": torch.mean(adversarial_losses).item(),
        }
        print(eval_log)

        if self.wandb:
            self.wandb.log(eval_log)

        return torch.mean(style_losses)

    def get_biber_losses(self, batch, biber_encoding):
        outputs = self.model(**batch.to(self.device), output_hidden_states=True)
        cls = self._get_cls(outputs, training=False)
        style_loss = self.biber_style_loss(cls, biber_encoding)
        adversarial_loss = self.biber_adversarial_loss(cls, biber_encoding)
        return style_loss, adversarial_loss

    def _get_cls(self, outputs, training=True):
        last_hidden_state = outputs["hidden_states"][-1]
        cls_representation = last_hidden_state[:, 0, :]
        if training:
            return self.cls_dropout(cls_representation)
        return cls_representation

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
