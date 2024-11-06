import os
import sys
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback

import wandb
import logging

sys.path.append('../..')

from modeling.tagger.train_utils import StreamingDataset, compute_metrics, compute_tag_level_metrics, EvalSamplingTrainer

# Path Configuration
BASE_DIR = "/shared/3/projects/hiatus/tagged_data/models"
MODEL_NAME = "roberta-base"
RUN_NAME = "binary-finetune-full"
NUM_EPOCHS = 8
BATCH_SIZE = 96

RUN_DIR = os.path.join(BASE_DIR, MODEL_NAME, RUN_NAME)
OUTPUT_DIR = os.path.join(RUN_DIR, "results")
WANDB_RUN_NAME = f"{MODEL_NAME}-{RUN_NAME}"
MODEL_SAVE_PATH = os.path.join(RUN_DIR, "best_model")
TAG_PERFORMANCE_SUMMARY_PATH = os.path.join(RUN_DIR, "tag_level_performance_summary.csv")

# Create necessary directories
try:
    os.makedirs(RUN_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
except Exception as e:
    print(f"Failed to create directories: {e}")
    raise

# Add logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

logger.info(f"Starting training run with configuration:")
logger.info(f"Model: {MODEL_NAME}")
logger.info(f"Run name: {RUN_NAME}")
logger.info(f"Output directory: {OUTPUT_DIR}")
logger.info(f"Number of epochs: {NUM_EPOCHS}")


# ============================================================================
# Data Loading and Preprocessing
# ============================================================================

logger.info(f"Loading tokenizer and model from {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

logger.info("Initializing streaming datasets...")
train_dataset = StreamingDataset('/shared/3/projects/hiatus/tagged_data/binary_train.jsonl', tokenizer, max_length=512)
dev_dataset = StreamingDataset('/shared/3/projects/hiatus/tagged_data/binary_dev.jsonl', tokenizer, max_length=512)
test_dataset = StreamingDataset('/shared/3/projects/hiatus/tagged_data/binary_test.jsonl', tokenizer, max_length=512)

logger.info(f"Loading tokenizer and model from {MODEL_NAME}...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    problem_type="regression",
    num_labels=len(train_dataset.label_columns)
).to('cuda')

logger.info(f"Dataset sizes - Train: {train_dataset.total_rows}, Dev: {dev_dataset.total_rows}, Test: {test_dataset.total_rows}")
logger.info(f"Number of labels: {len(train_dataset.label_columns)}")
logger.info(f"Labels: {', '.join(train_dataset.label_columns)}")

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
        "batch_size": BATCH_SIZE,
        "weight_decay": 0.01,
        "num_labels": len(train_dataset.label_columns),
        "labels": train_dataset.label_columns,
        "train_size": train_dataset.total_rows,
        "dev_size": dev_dataset.total_rows,
        "test_size": test_dataset.total_rows
    }
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    evaluation_strategy="steps",
    eval_steps=0.25,
    save_strategy="steps",
    save_steps=0.25,
    learning_rate=2e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    weight_decay=0.01,
    push_to_hub=False,
    load_best_model_at_end=True,
    report_to="wandb",
    metric_for_best_model="eval_mse",
    greater_is_better=True,
    fp16=True
)

trainer = EvalSamplingTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=5)
    ]
)

train_result = trainer.train()
logger.info(f"Training completed. Metrics: {train_result.metrics}")
wandb.log({"training_loss": train_result.metrics["train_loss"]})

logger.info("Evaluating on test set...")
test_results = trainer.evaluate(test_dataset)
logger.info(f"Test results: {test_results}")
wandb.log({
    "test_mse": test_results["eval_mse"],
    "test_rmse": test_results["eval_rmse"],
    "test_mae": test_results["eval_mae"],
    "test_r2": test_results["eval_r2"],
    "test_precision_micro": test_results["eval_precision_micro"],
    "test_recall_micro": test_results["eval_recall_micro"],
    "test_f1_micro": test_results["eval_f1_micro"],
    "test_precision_macro": test_results["eval_precision_macro"],
    "test_recall_macro": test_results["eval_recall_macro"],
    "test_f1_macro": test_results["eval_f1_macro"]
})

logger.info(f"Saving model to {MODEL_SAVE_PATH}")
trainer.save_model(MODEL_SAVE_PATH)

# ============================================================================
# Results Analysis and Logging
# ============================================================================

logger.info("Computing tag-level metrics...")
test_predictions = trainer.predict(test_dataset)
tag_level_metrics = compute_tag_level_metrics(
    test_predictions.predictions,
    test_predictions.label_ids,
    train_dataset.label_columns
)

# Log detailed results
logger.info("\nTag-level performance on test set:")
for tag, metrics in tag_level_metrics.items():
    logger.info(f"\n{tag}:")
    for metric_name, metric_value in metrics.items():
        logger.info(f"  {metric_name}: {metric_value:.4f}")
        wandb.log({f"{tag}_{metric_name}": metric_value})

# Calculate and log overall statistics
avg_metrics = {
    # 'avg_mse': np.mean([m['mse'] for m in tag_level_metrics.values()]),
    # 'avg_rmse': np.mean([m['rmse'] for m in tag_level_metrics.values()]),
    # 'avg_mae': np.mean([m['mae'] for m in tag_level_metrics.values()]),
    # 'avg_r2': np.mean([m['r2'] for m in tag_level_metrics.values()]),
    'avg_precision': np.mean([m['precision'] for m in tag_level_metrics.values()]),
    'avg_recall': np.mean([m['recall'] for m in tag_level_metrics.values()]),
    'avg_f1': np.mean([m['f1'] for m in tag_level_metrics.values()])
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
        'MSE': metrics['mse'],
        'RMSE': metrics['rmse'],
        'MAE': metrics['mae'],
        'R2': metrics['r2'],
        'Precision': metrics['precision'],
        'Recall': metrics['recall'],
        'F1': metrics['f1']
    }
    for tag, metrics in tag_level_metrics.items()
]

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(TAG_PERFORMANCE_SUMMARY_PATH, index=False)

logger.info("Training run completed successfully")
wandb.finish()