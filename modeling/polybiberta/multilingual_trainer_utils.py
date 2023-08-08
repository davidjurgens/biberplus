import sys

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.append('../')

from src.custom_training.multilingual_finetuning.collators import MLMTextBiberCollator, MLMTextCollator


def alternate_loaders(multilingual_dataloader, english_dataloader):
    # Initialize the iterators for both dataloaders
    multilingual_dataloader_iter = iter(multilingual_dataloader)
    english_dataloader_iter = iter(english_dataloader)
    print(f"Loading alternate data loaders. Multilingual batches: {len(multilingual_dataloader)} \t English batches: "
          f"{len(english_dataloader)}")
    assert len(multilingual_dataloader) > len(english_dataloader)

    total_steps = len(multilingual_dataloader) + len(english_dataloader)
    alternate_index = round(total_steps / len(english_dataloader))

    for i in range(total_steps):
        if (i + 1) % alternate_index == 0:
            batch, english_dataloader_iter = get_next_batch(english_dataloader, english_dataloader_iter)
            yield batch, 'english'
        else:
            batch, multilingual_dataloader_iter = get_next_batch(multilingual_dataloader, multilingual_dataloader_iter)
            yield batch, 'multilingual'


def get_next_batch(dataloader, iterator):
    try:
        batch = next(iterator)
    except StopIteration:
        # Reinitialize the iterator if we've exhausted the dataloader
        iterator = iter(dataloader)
        batch = next(iterator)
    return batch, iterator


def load_all_dataloaders(args):
    tokenizer = load_tokenizer(args.tokenizer if args.tokenizer else args.pretrained_model)

    collator = MLMTextCollator(tokenizer=tokenizer, max_length=args.max_length)
    en_collator = MLMTextBiberCollator(tokenizer=tokenizer, max_length=args.max_length)

    train_loader, dev_loader = get_dataloaders(args, collator)
    en_train_loader, en_dev_loader = get_dataloaders(args, en_collator, is_english=True)

    return train_loader, dev_loader, en_train_loader, en_dev_loader


def get_dataloaders(args, collator, is_english=False):
    print(
        f"Loading in the {'English' if is_english else 'multilingual'} training data from {args.en_train_file if is_english else args.train_file}")
    train = load_dataset("json", data_files=args.en_train_file if is_english else args.train_file, split="train")
    dev = load_dataset("json", data_files=args.en_dev_file if is_english else args.dev_file, split="train")
    if args.num_training_samples > 1:
        train = train.select(range(args.num_training_samples))
    if args.num_eval_samples > 1:
        dev = dev.select(range(args.num_eval_samples))

    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    dev_loader = DataLoader(dev, batch_size=args.batch_size, shuffle=True, collate_fn=collator)

    return train_loader, dev_loader


def load_tokenizer(pretrained_model):
    print(f"Loading in {pretrained_model} tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
