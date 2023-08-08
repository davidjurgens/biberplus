import os
import sys

import torch
from accelerate import Accelerator
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup, AutoModelForMaskedLM

sys.path.append('../')
from modeling.contrastive_training.evaluation import evaluate


def load_model(checkpoint_dir):
    model = AutoModelForMaskedLM.from_pretrained(checkpoint_dir)

    optimizer_and_scheduler = torch.load(os.path.join(checkpoint_dir, 'optimizer_and_scheduler.pt'),
                                         map_location='cuda:0')

    optimizer = AdamW(model.parameters())
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=0)

    optimizer.load_state_dict(optimizer_and_scheduler['optimizer_state_dict'])
    scheduler.load_state_dict(optimizer_and_scheduler['scheduler_state_dict'])

    return model, optimizer, scheduler


def calculate_perplexity(mlm_losses):
    try:
        return torch.exp(torch.mean(mlm_losses)).item()
    except OverflowError:
        return float("inf")


def encode_batches(model, batchA, batchB, device):
    z1 = model(**batchA.to(device)).pooler_output
    z2 = model(**batchB.to(device)).pooler_output
    return z1, z2


def evaluate_and_save_model(model, dev_loader, args, best_perf):
    results = evaluate(model, dev_loader, args)
    return save_checkpoints(model, args, results, best_perf)


def save_checkpoints(encoder_model, args, results, best_perf):
    if hasattr(encoder_model, 'save_pretrained'):
        encoder_model.save_pretrained(args.out_dir + "/last_model")
    else:
        torch.save(encoder_model.state_dict(), args.out_dir + "/last_model/pytorch_model.pth")
    if best_perf < results['MRR']:
        if hasattr(encoder_model, 'save_pretrained'):
            encoder_model.save_pretrained(args.out_dir + "/best_model")
        else:
            torch.save(encoder_model.state_dict(), args.out_dir + "/best_model/pytorch_model.pth")
        best_perf = results['MRR']

    return best_perf


def setup_accelerator(model, train_loader, args):
    # use HuggingFace accelerator to manage multi-gpu training
    accelerator = Accelerator(gradient_accumulation_steps=args.grad_acc)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.num_warmup_steps,
                                                num_training_steps=len(train_loader) * args.epochs, num_cycles=1)

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )
    return accelerator, model, optimizer, train_dataloader, scheduler
