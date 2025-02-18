import pandas as pd
import json
from tqdm import tqdm
import os

import sys

sys.path.append('../..')

from modeling.neurobiber.tagger import load_model_and_tokenizer, predict_batch

# Input/Output paths
INPUT_PATH = '/shared/3/projects/hiatus/core-corpus/train.tsv'
OUTPUT_DIR = os.path.dirname(INPUT_PATH)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'neurobiber_train.jsonl')

# Load model and tokenizer
model, tokenizer = load_model_and_tokenizer()

print(f"\nWill save predictions to: {OUTPUT_FILE}")


# Read and sample data
df = pd.read_csv(INPUT_PATH, 
                 sep='\t',
                 names=['register', 'doc_id', 'text'])

df = df.dropna(subset=['register'])

print(f"Processing {len(df)} documents...\n")

def validate_text(text):
    if not isinstance(text, str):
        return str(text)  # Convert to string if possible
    return text

# Process in batches
BATCH_SIZE = 128
processed_count = 0
error_count = 0

with open(OUTPUT_FILE, 'w') as f:
    for i in tqdm(range(0, len(df), BATCH_SIZE)):
        batch = df.iloc[i:i+BATCH_SIZE]
        
        try:
            texts = [validate_text(text) for text in batch['text'].tolist()]
            
            # Add length check for empty texts
            if not any(texts):
                print(f"Skipping batch at index {i} - all empty texts")
                continue
                
            predictions = predict_batch(model, tokenizer, texts)
            
            # Verify predictions shape matches batch size
            if len(predictions) != len(batch):
                print(f"Warning: Prediction count mismatch at batch {i}")
                continue
            
            # Write results
            for batch_idx, row in enumerate(batch.itertuples()):
                try:
                    json_obj = {
                        'doc_id': row.doc_id,
                        'text': row.text,
                        'register': row.register.split(),
                        'neural_tags': predictions[batch_idx].tolist()
                    }
                    f.write(json.dumps(json_obj) + '\n')
                    processed_count += 1
                except Exception as e:
                    error_count += 1
                    print(f"Error processing row {row.doc_id}: {str(e)}")
                    continue
                    
        except Exception as e:
            error_count += 1
            print(f"Error processing batch starting at index {i}: {str(e)}")
            continue

print(f"Created {OUTPUT_FILE}")
print(f"Successfully processed: {processed_count} samples")
if error_count > 0:
    print(f"Errors encountered: {error_count}")
