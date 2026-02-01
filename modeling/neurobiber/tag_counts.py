import json
import pandas as pd

from tqdm import tqdm

train_fp = "/shared/3/projects/hiatus/tagged_data/biber-aggregate/binary_train.jsonl"
dev_fp = "/shared/3/projects/hiatus/tagged_data/biber-aggregate/binary_dev.jsonl"
test_fp = "/shared/3/projects/hiatus/tagged_data/biber-aggregate/binary_test.jsonl"

output_fp = "/shared/3/projects/hiatus/tagged_data/biber-aggregate/tag_counts.csv"


def process_jsonl(filepath):
    # Initialize counter dictionary
    tag_counts = {}

    # Process file line by line
    with open(filepath, "r") as f:
        for line in tqdm(f, desc=f"Processing {filepath.split('/')[-1]}"):
            # Parse single JSON line
            data = json.loads(line)
            features = data["features"]

            # Count each feature that has a non-zero mean
            for key, value in features.items():
                if key.endswith("_mean") and value > 0:
                    # Remove '_mean' suffix and increment counter
                    tag_name = key.replace("_mean", "")
                    tag_counts[tag_name] = tag_counts.get(tag_name, 0) + 1

    return tag_counts


# Process each split
splits = {"train": train_fp, "dev": dev_fp, "test": test_fp}

# Get counts for each split
results = {}
for split_name, filepath in splits.items():
    results[split_name] = process_jsonl(filepath)

# Convert to DataFrame
results_df = pd.DataFrame(results)
# Add tag names as a column instead of just being the index
results_df.reset_index(inplace=True)
results_df.rename(columns={"index": "tag"}, inplace=True)

results_df.to_csv(output_fp, index=False)
