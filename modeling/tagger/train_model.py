import os
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
import wandb
from typing import Dict, List
import logging

# ============================================================================
# Configuration and Setup
# ============================================================================

# GPU Settings
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Path Configuration
BASE_DIR = "/shared/3/projects/hiatus/tagged_data/models"
MODEL_NAME = "roberta-base"
RUN_NAME = "finetune"
NUM_EPOCHS = 5

RUN_DIR = os.path.join(BASE_DIR, MODEL_NAME, RUN_NAME)
OUTPUT_DIR = os.path.join(RUN_DIR, "results")
WANDB_RUN_NAME = f"{MODEL_NAME}-{RUN_NAME}"
MODEL_SAVE_PATH = os.path.join(RUN_DIR, "best_model")
TAG_PERFORMANCE_SUMMARY_PATH = os.path.join(RUN_DIR, "tag_level_performance_summary.csv")

# Create necessary directories
os.makedirs(RUN_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# Add logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add config logging
logger.info(f"Starting training run with configuration:")
logger.info(f"Model: {MODEL_NAME}")
logger.info(f"Run name: {RUN_NAME}")
logger.info(f"Output directory: {OUTPUT_DIR}")
logger.info(f"Number of epochs: {NUM_EPOCHS}")

# ============================================================================
# Data Loading and Preprocessing
# ============================================================================

logger.info("Loading datasets...")
train_df = pd.read_csv('/shared/3/projects/hiatus/tagged_data/binary_train.tsv', sep='\t')
dev_df = pd.read_csv('/shared/3/projects/hiatus/tagged_data/binary_dev.tsv',  sep='\t')
test_df = pd.read_csv('/shared/3/projects/hiatus/tagged_data/binary_test.tsv',  sep='\t')

text_column = 'text'
label_columns = train_df.columns[1:].tolist()

logger.info(f"Dataset sizes - Train: {len(train_df)}, Dev: {len(dev_df)}, Test: {len(test_df)}")
logger.info(f"Number of labels: {len(label_columns)}")
logger.info(f"Labels: {', '.join(label_columns)}")

# Convert to HuggingFace datasets
train_dataset = Dataset.from_pandas(train_df)
dev_dataset = Dataset.from_pandas(dev_df)
test_dataset = Dataset.from_pandas(test_df)

# ============================================================================
# Model and Tokenizer Setup
# ============================================================================

logger.info(f"Loading tokenizer and model from {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    problem_type="multi_label_classification",
    num_labels=len(label_columns)
).to('cuda')

def tokenize_function(examples):
    return tokenizer(examples[text_column], padding="max_length", truncation=True, max_length=512)

logger.info("Tokenizing datasets...")
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_dev = dev_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)
logger.info("Tokenization complete")

# ============================================================================
# Metrics and Evaluation Functions
# ============================================================================

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.array(predictions >= 0.5, dtype=float)
    
    f1_micro = f1_score(labels, predictions, average='micro')
    f1_macro = f1_score(labels, predictions, average='macro')
    roc_auc = roc_auc_score(labels, predictions, average='macro', multi_class='ovr')
    
    return {
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'roc_auc': roc_auc
    }

def compute_tag_level_metrics(predictions: np.ndarray, labels: np.ndarray, tag_names: List[str]) -> Dict[str, Dict[str, float]]:
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

# ============================================================================
# Training Setup and Execution
# ============================================================================

logger.info("Initializing wandb...")
wandb.init(
    project="biber-multidimensional-register-analysis",
    name=RUN_NAME,
    config={
        "model_name": MODEL_NAME,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": 2e-5,
        "batch_size": 16,
        "weight_decay": 0.01,
        "num_labels": len(label_columns),
        "labels": label_columns,
        "train_size": len(train_df),
        "dev_size": len(dev_df),
        "test_size": len(test_df)
    }
)

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=0.01,
    push_to_hub=False,
    load_best_model_at_end=True,
    report_to="wandb",
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_dev,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[
        transformers.EarlyStoppingCallback(early_stopping_patience=3)
    ]
)

# Training progress will be logged automatically by the Trainer
train_result = trainer.train()
logger.info(f"Training completed. Metrics: {train_result.metrics}")
wandb.log({"training_loss": train_result.metrics["train_loss"]})

logger.info("Evaluating on test set...")
test_results = trainer.evaluate(tokenized_test)
logger.info(f"Test results: {test_results}")
wandb.log({
    "test_loss": test_results["eval_loss"],
    "test_f1_micro": test_results["eval_f1_micro"],
    "test_f1_macro": test_results["eval_f1_macro"],
    "test_roc_auc": test_results["eval_roc_auc"]
})

logger.info(f"Saving model to {MODEL_SAVE_PATH}")
trainer.save_model(MODEL_SAVE_PATH)

# ============================================================================
# Results Analysis and Logging
# ============================================================================

logger.info("Computing tag-level metrics...")
test_predictions = trainer.predict(tokenized_test)
test_predictions_binary = np.array(test_predictions.predictions >= 0.5, dtype=float)
tag_level_metrics = compute_tag_level_metrics(test_predictions_binary, test_predictions.label_ids, label_columns)

# Log detailed results
logger.info("\nTag-level performance on test set:")
for tag, metrics in tag_level_metrics.items():
    logger.info(f"\n{tag}:")
    for metric_name, metric_value in metrics.items():
        logger.info(f"  {metric_name}: {metric_value:.4f}")
        wandb.log({f"{tag}_{metric_name}": metric_value})

# Calculate and log overall statistics
avg_metrics = {
    'avg_precision': np.mean([m['precision'] for m in tag_level_metrics.values()]),
    'avg_recall': np.mean([m['recall'] for m in tag_level_metrics.values()]),
    'avg_f1': np.mean([m['f1'] for m in tag_level_metrics.values()]),
    'avg_auc': np.mean([m['auc'] for m in tag_level_metrics.values()])
}

logger.info("\nOverall Statistics:")
for metric_name, value in avg_metrics.items():
    logger.info(f"{metric_name}: {value:.4f}")
    wandb.log({metric_name: value})

# Save summary to CSV
logger.info(f"Saving performance summary to {TAG_PERFORMANCE_SUMMARY_PATH}")
summary_data = [
    {
        'Tag': tag,
        'Precision': metrics['precision'],
        'Recall': metrics['recall'],
        'F1 Score': metrics['f1'],
        'AUC': metrics['auc']
    }
    for tag, metrics in tag_level_metrics.items()
]

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(TAG_PERFORMANCE_SUMMARY_PATH, index=False)

logger.info("Training run completed successfully")
wandb.finish()