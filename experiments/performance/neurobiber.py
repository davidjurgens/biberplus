import sys
import time

sys.path.append("/home/kalkiek/projects/biber-multidimensional-register-analysis/")

from experiments.performance.data_utils import load_saved_dataset
from modeling.neurobiber.tagger import (
    load_model_and_tokenizer,
    predict_batch,
)


def process_texts_neurobiber(texts, model, tokenizer, batch_size=128):
    start_time = time.time()

    # Use predict_batch which internally handles chunking
    predictions = predict_batch(model, tokenizer, texts)

    processing_time = time.time() - start_time
    return predictions, processing_time


def main():
    # Load dataset
    texts = load_saved_dataset()
    print(f"\nLoaded {len(texts)} texts")

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    print("Model and tokenizer loaded")

    # Process texts and measure time
    results, processing_time = process_texts_neurobiber(texts, model, tokenizer)

    # Print performance metrics
    print(f"\nProcessing completed in {processing_time:.3f} seconds")
    print(f"Average time per text: {processing_time / len(texts):.3f} seconds")


if __name__ == "__main__":
    main()
