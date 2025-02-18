import os
import sys
import time
import json
import numpy as np 
import pandas as pd
import pickle
from tqdm import tqdm

def load_samples(dataset_path, n_samples):
    """Load first n_samples from a corpus.jsonl file"""
    texts = []
    with open(os.path.join(dataset_path, "corpus.jsonl"), 'r') as f:
        for i, line in enumerate(f):
            if i >= n_samples:
                break
            sample = json.loads(line)
            texts.append(sample['fullText'])
    return texts

def load_all_datasets(data_dir, datasets, n_samples):
    """Load samples from multiple datasets"""
    all_texts = []
    for dataset in tqdm(datasets, desc="Loading datasets"):
        dataset_path = os.path.join(data_dir, dataset)
        texts = load_samples(dataset_path, n_samples)
        all_texts.extend(texts)
    return all_texts

def calculate_statistics(texts):
    """Calculate token length statistics for a list of texts"""
    token_lengths = [len(text.split()) for text in texts]
    stats = {
        'mean': np.mean(token_lengths),
        'min': np.min(token_lengths),
        'max': np.max(token_lengths),
        'median': np.median(token_lengths),
        'std': np.std(token_lengths),
        'total': sum(token_lengths)
    }
    return stats

def save_dataset(texts, filename='dataset.pkl'):
    """Save the loaded texts to a pickle file"""
    with open(filename, 'wb') as f:
        pickle.dump(texts, f)

def load_saved_dataset(filename='dataset.pkl'):
    """Load the saved texts from a pickle file"""
    with open(filename, 'rb') as f:
        return pickle.load(f)

def main():
    # Constants moved into main
    data_dir = "/shared/3/projects/hiatus/tagged_data"
    datasets = [
        'amazon',
        'reddit',
        'gmane',
        'book3corpus',
        'ao3',
        'wiki',
        'wiki_discussions'
    ]
    n_samples = 1000
    
    all_texts = load_all_datasets(data_dir, datasets, n_samples)
    
    # Save dataset for future use
    save_dataset(all_texts)
    
    # Calculate and print statistics
    stats = calculate_statistics(all_texts)
    print(f"Total tokens: {stats['total']:,}")
    print(f"Average token length: {stats['mean']:.2f} tokens")
    print(f"\nDistribution statistics:")
    print(f"Min: {stats['min']}")
    print(f"Max: {stats['max']}")
    print(f"Median: {stats['median']}")
    print(f"Std dev: {stats['std']:.2f}")

if __name__ == "__main__":
    main()