import os
import sys
import time
import json
import numpy as np 
import pandas as pd

sys.path.append('/home/kalkiek/projects/biber-multidimensional-register-analysis/')

from tqdm import tqdm
from experiments.performance.data_utils import load_saved_dataset
from biberplus.tagger import load_config, load_pipeline, calculate_tag_frequencies, tag_text

def process_texts(texts, pipeline, config):
    """Process all texts and return frequencies"""
    start_time = time.time()
    
    results = []
    for text in tqdm(texts, desc="Processing texts"):
        # tags = tag_text(text, pipeline)
        if text:
            frequencies = calculate_tag_frequencies(text, pipeline, config)
            results.append(frequencies)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    return results, processing_time

def main():
    # Load configuration
    config = load_config()
    config.update({
        'use_gpu': False, 
        'n_processes': 4, 
        'function_words': False,
        'token_normalization': 100,
    })
    pipeline = load_pipeline(config)
    
    print("Configuration loaded:")
    print(json.dumps(config, indent=2))
    
    # Load dataset
    texts = load_saved_dataset()
    print(f"\nLoaded {len(texts)} texts")
    
    # Process texts and measure time
    results, processing_time = process_texts(texts, pipeline, config)
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Print performance metrics
    print(f"\nProcessing completed in {processing_time:.2f} seconds")
    print(f"Average time per text: {processing_time/len(texts):.2f} seconds")
    
    # Optional: save results
    df.to_csv('biber_results.csv', index=False)

if __name__ == "__main__":
    main()