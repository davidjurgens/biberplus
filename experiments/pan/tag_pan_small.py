import sys
import pandas as pd
from tqdm import tqdm
import os

sys.path.append("../..")

from modeling.neurobiber.tagger import load_model_and_tokenizer, get_predictions

# Constants
input_directory = "/shared/3/datasets/PAN/pan20-av-training-small/"
output_directory = os.path.join(input_directory, "tagged")

# Create output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Load model and tokenizer
model, tokenizer = load_model_and_tokenizer()

# Read datasets
train = pd.read_json(input_directory + "train.jsonl", lines=True)
dev = pd.read_json(input_directory + "dev.jsonl", lines=True)
test = pd.read_json(input_directory + "test.jsonl", lines=True)
print(f"Dataset sizes - Train: {len(train)}, Dev: {len(dev)}, Test: {len(test)}")

for name, df in [("train", train), ("dev", dev), ("test", test)]:
    print(f"\nProcessing {name} set...")

    # Process first text in pair
    tqdm.pandas(desc=f"Tagging {name} texts (1/2)")
    texts_1 = df["pair"].apply(lambda x: x[0]).tolist()
    predictions_1 = get_predictions(model, texts_1, tokenizer)
    df["neural_biber_1"] = list(predictions_1)

    # Process second text in pair
    tqdm.pandas(desc=f"Tagging {name} texts (2/2)")
    texts_2 = df["pair"].apply(lambda x: x[1]).tolist()
    predictions_2 = get_predictions(model, texts_2, tokenizer)
    df["neural_biber_2"] = list(predictions_2)

    # Save processed dataset
    output_path = os.path.join(output_directory, f"{name}_tagged.jsonl")
    df.to_json(output_path, orient="records", lines=True)
    print(f"Saved {name} set to {output_path}")
