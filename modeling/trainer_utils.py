import os
import sys

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

sys.path.append('../')

from modeling.biberta.collator import MLMBiberPairCollator, MLMBiberCollator
from modeling.contrastive_training.collator import ContrastiveBiberCollator, ContrastiveCollator


def load_model_tokenizer(pretrained_model='roberta-base', gradient_checkpointing=False):
    print(f"Loading in {pretrained_model} model")
    model = AutoModel.from_pretrained(pretrained_model)

    if gradient_checkpointing:
        model.encoder.gradient_checkpointing = True

    print(f"Loading in {pretrained_model} tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def load_tokenizer(pretrained_model):
    print(f"Loading in {pretrained_model} tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_dataloaders(args, pairs=True, mlm=False, biber=False):
    tokenizer = load_tokenizer(args.tokenizer if args.tokenizer else args.pretrained_model)
    train_dataset, dev_dataset = load_datasets(args)
    collator, eval_collator = get_collators(tokenizer, args.max_length, mlm, pairs, biber)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    eval_dataloader = DataLoader(dev_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=eval_collator)

    return train_dataloader, eval_dataloader


def get_collators(tokenizer, max_length, mlm=False, pairs=True, biber=False):
    if mlm:
        CollatorClass = MLMBiberPairCollator if pairs else MLMBiberCollator
    else:
        CollatorClass = ContrastiveBiberCollator if biber else ContrastiveCollator

    return create_collators(CollatorClass, tokenizer, max_length)


def create_collators(CollatorClass, tokenizer, max_length):
    collator = CollatorClass(tokenizer=tokenizer, max_length=max_length)
    eval_collator = CollatorClass(tokenizer=tokenizer, max_length=max_length, evaluate=True)
    return collator, eval_collator


def load_datasets(args):
    print(f"Reading in train data from {args.train_data}")
    train_dataset = load_dataset("json", data_files=args.train_data, split="train")
    print(f"Reading in evaluation data from {args.dev_data}")
    dev_dataset = load_dataset("json", data_files=args.dev_data, split="train").shuffle(seed=42)

    if args.num_training_samples > 1:
        train_dataset = train_dataset.select(range(args.num_training_samples))
    if args.num_eval_samples > 1:
        dev_dataset = dev_dataset.select(range(args.num_eval_samples))

    return train_dataset, dev_dataset


def make_output_dirs(out_dir):
    print("Creating output directories in " + out_dir)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if not os.path.exists(out_dir + "/last_model"):
        os.mkdir(out_dir + "/last_model")
    if not os.path.exists(out_dir + "/best_model"):
        os.mkdir(out_dir + "/best_model")
