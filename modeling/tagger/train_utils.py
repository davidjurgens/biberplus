import numpy as np
import pandas as pd

from torch.utils.data import IterableDataset
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

def chunk_df_iterator(file_path, chunk_size=1000):
    try:
        for chunk in pd.read_csv(file_path, sep='\t', chunksize=chunk_size):
            yield chunk
    except Exception as e:
        raise RuntimeError(f"Error reading file {file_path}: {str(e)}")

class StreamingDataset(IterableDataset):
    def __init__(self, file_path, tokenizer, chunk_size=1000, batch_size=16):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        
        # Get number of rows and columns without loading entire file
        try:
            with pd.read_csv(file_path, sep='\t', chunksize=1) as reader:
                first_chunk = next(reader)
                self.label_columns = first_chunk.columns[1:].tolist()
                
                # More efficient row counting using wc -l if on Unix, or falling back to chunk counting
                try:
                    import subprocess
                    result = subprocess.run(['wc', '-l', file_path], capture_output=True, text=True)
                    self.total_rows = int(result.stdout.split()[0]) - 1  # -1 for header
                except:
                    self.total_rows = sum(1 for _ in pd.read_csv(file_path, sep='\t', usecols=[0]))
        except Exception as e:
            raise RuntimeError(f"Failed to initialize dataset from {file_path}: {str(e)}")
        
        self.batch_size = batch_size
        
    def get_total_steps(self):
        return self.total_rows // self.batch_size
        
    def __len__(self):
        return self.get_total_steps()
        
    def __iter__(self):
        for chunk in chunk_df_iterator(self.file_path, self.chunk_size):
            texts = chunk['text'].tolist()
            labels = chunk[self.label_columns].values.tolist()
            
            # Tokenize texts
            encodings = self.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # Yield individual examples
            for i in range(len(texts)):
                yield {
                    'input_ids': encodings['input_ids'][i],
                    'attention_mask': encodings['attention_mask'][i],
                    'labels': labels[i]
                }

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.array(predictions >= 0.5, dtype=float)
    
    metrics = {
        'f1_micro': f1_score(labels, predictions, average='micro'),
        'f1_macro': f1_score(labels, predictions, average='macro')
    }
    
    try:
        metrics['roc_auc'] = roc_auc_score(labels, predictions, average='macro', multi_class='ovr')
    except ValueError as e:
        print(f"Could not compute ROC AUC: {e}")
        metrics['roc_auc'] = float('nan')
    
    return metrics

def compute_tag_level_metrics(predictions, labels, tag_names):
    tag_metrics = {}
    for i, tag in enumerate(tag_names):
        tag_predictions = predictions[:, i]
        tag_labels = labels[:, i]
        tag_metrics[tag] = {
            'precision': precision_score(tag_labels, tag_predictions),
            'recall': recall_score(tag_labels, tag_predictions),
            'f1': f1_score(tag_labels, tag_predictions),
            'auc': roc_auc_score(tag_labels, tag_predictions)
        }
    return tag_metrics