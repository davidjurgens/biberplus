import sys
import argparse
import wandb

sys.path.append("../../..")

from src.custom_training.contrastive_training.models import (
    ContrastiveModel,
    StyleContrastiveModel,
)


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_data",
        default="/shared/3/datasets/PAN/pan20-av-training-small/contrastive/train.jsonl",
        type=str,
    )
    parser.add_argument(
        "--dev_data",
        default="/shared/3/datasets/PAN/pan20-av-training-small/contrastive/dev.jsonl",
        type=str,
    )
    parser.add_argument(
        "--out_dir", type=str, default="/shared/3/projects/hiatus/models/pan20/"
    )
    parser.add_argument("--pretrained_model", default="roberta-base", type=str)
    parser.add_argument("--style_training", action="store_true")
    parser.add_argument("--tokenizer", default=None, type=str)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--grad_norm", default=1.0, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--eval_batch_size", default=128, type=int)
    parser.add_argument("--grad_acc", default=1, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--saving_step", default=200, type=int)
    parser.add_argument("--weight_decay", default=1e-2, type=float)
    parser.add_argument("--max_length", default=350, type=int)
    parser.add_argument("--gradient_checkpointing", default=False, type=bool)
    parser.add_argument("--num_warmup_steps", default=1000, type=int)
    parser.add_argument("--num_training_samples", default=-1, type=int)
    parser.add_argument("--num_eval_samples", default=-1, type=int)
    parser.add_argument("--evaluate", action="store_true")
    # the following arguments are only relevant if you hope to log results in wandb
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--entity", default="sadiri-michigan", type=str)
    parser.add_argument("--project_name", default="pan20", type=str)
    parser.add_argument("--run_name", default="V1", type=str)
    parser.add_argument(
        "--code_dir", default="src/custom_training/contrastive_training/", type=str
    )

    return parser.parse_args()


def setup_wandb(args):
    wandb.init(
        project=args.project_name,
        entity="sadiri-michigan",
        settings=wandb.Settings(code_dir=args.code_dir),
        config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "max_length": args.max_length,
            "saving_step": args.saving_step,
            "grad_norm": args.grad_norm,
            "learning_rate": args.learning_rate,
            "pretrained_model": args.pretrained_model,
            "gradient_accumulation": args.grad_acc,
        },
    )
    wandb.run.name = args.run_name
    return wandb


if __name__ == "__main__":
    args = argument_parser()

    wb = setup_wandb(args) if args.wandb else None

    if args.style_training:
        model = StyleContrastiveModel(args, wandb=wb)
    else:
        model = ContrastiveModel(args, wandb=wb)

    model.train_model()
    model.evaluate()
    model.save_model(step=str(-1), version="last")
