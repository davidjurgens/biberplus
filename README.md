# Biberplus

[![PyPI version](https://img.shields.io/pypi/v/biberplus.svg?style=flat)](https://pypi.org/project/biberplus/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Biberplus is a pure Python implementation of the linguistic tagging system introduced in Biber (1988). Built upon the spaCy library, it delivers fast part-of-speech tagging along with supplemental features such as a function word tagger, PCA, and factor analysis. These features, inspired by the work of Grieve, Clarke, and colleagues, make Biberplus a powerful tool for analyzing large text corpora.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quickstart Guide](#quickstart-guide)
  - [Biber Tagger](#1-biber-tagger)
  - [Function Words Tagger](#2-function-words-tagger)
  - [Text Embeddings](#3-text-embeddings)
  - [Dimension Reduction](#4-dimension-reduction)
- [Configuration](#configuration)
- [Usage Tips](#usage-tips)
- [Troubleshooting](#troubleshooting)
- [References](#references)
- [License](#license)

---

## Features

- **Linguistic Tagging:** Implements Biber's tags alongside supplementary features.
- **Function Word Analysis:** Option to use built-in or custom lists for function word tagging.
- **Text Embeddings:** Flatten tagging frequencies into a vector representation.
- **Dimension Reduction:** Perform PCA and factor analysis on the resulting data.
- **Performance:** Support for multi-processing and GPU acceleration.

---

## Installation

### From PyPI (Stable Release)
Install the latest version (0.3.0) from PyPI:

```bash
pip install biberplus
```

For more details and package history, visit the [Biberplus project page on PyPI](https://pypi.org/project/biberplus/0.3.0/).

**Important:**  
Biberplus depends on spaCy for text processing. After installing biberplus, you must manually download the spaCy English model by running:

```bash
python -m spacy download en_core_web_sm
```

---

## Quickstart Guide

### 1. Biber Tagger

**Tag a string using the default configuration:**
```python
from biberplus.tagger import calculate_tag_frequencies

frequencies_df = calculate_tag_frequencies("Your sample text goes here")
print(frequencies_df)
```

**Tag a large corpus with GPU and multi-processing:**
```python
from biberplus.tagger import load_config, load_pipeline, calculate_tag_frequencies

config = load_config()
config.update({'use_gpu': True, 'n_processes': 4, 'function_words': False})
pipeline = load_pipeline(config)
frequencies_df = calculate_tag_frequencies("Your sample text goes here", pipeline, config)
print(frequencies_df)
```

### 2. Function Words Tagger

**Using the default list:**
```python
from biberplus.tagger import load_config, calculate_tag_frequencies

config = load_config()
config.update({'use_gpu': True, 'biber': False, 'function_words': True})
frequencies_df = calculate_tag_frequencies("Your sample text goes here")
print(frequencies_df)
```

**Using a custom list:**
```python
from biberplus.tagger import load_config, calculate_tag_frequencies

custom_fw = ["the", "of", "to", "and", "a", "in", "that"]
config = load_config()
config.update({
    'function_words': True,
    'biber': False,
    'grieve_clarke': False,
    'function_words_list': custom_fw
})
frequencies_df = calculate_tag_frequencies("Your sample text goes here", custom_fw)
print(frequencies_df)
```

### 3. Word-Level Tagging

See exactly which tags are applied to each word:
```python
import spacy
from biberplus.tagger import tag_text, load_config, load_pipeline

# Load configuration and pipeline
config = load_config()
pipeline = load_pipeline(config)

# Your test sentence
text = "It doesn't seem likely that this will work."

# Get tagged words
tagged_words = tag_text(text, pipeline=pipeline)

# Print each word and its tags
for word in tagged_words:
    print(f"Word: {word['text']:<15} Tags: {', '.join(word['tags'])}")
```

Example output:
```
Word: It              Tags: it, PIT, CAP, PRP, SBJP
Word: does            Tags: VPRT, SPAU
Word: n't             Tags: XX0, CONT, RB
Word: seem            Tags: SMP, INF
Word: likely          Tags: JJ
```

### 4. Text Embeddings

Generate an embedding vector from the textual data:
```python
from biberplus.tagger import load_config
from biberplus.reducer import encode_text

config = load_config()
embedding = encode_text(config, "Your sample text goes here")
print(embedding)
```

### 5. Dimension Reduction

**Using PCA:**
```python
from biberplus.tagger import load_config, load_pipeline, calculate_tag_frequencies
from biberplus.reducer import tags_pca

config = load_config()
config.update({'use_gpu': True, 'biber': True, 'function_words': True})
pipeline = load_pipeline(config)
frequencies_df = calculate_tag_frequencies("Your sample text goes here", pipeline, config)

pca_df, explained_variance = tags_pca(frequencies_df, components=2)
print(pca_df)
print(explained_variance)
```

---

## Configuration

The library uses a YAML configuration file located at `biberplus/tagger/config.yaml`. Common options include:

- `biber`: Enable Biber tag analysis.
- `function_words`: Enable function word tagging.
- `binary_tags`: Use binary features for tag counts.
- `token_normalization`: Number of tokens per batch for frequency calculation.
- `use_gpu`: Enable GPU acceleration.
- `n_processes`: Number of processes for multi-processing.
- `drop_last_batch_pct`: Drop the last batch if it is too small. What percentage counts as too small

You can modify these options in the file or update them dynamically in your script after loading the configuration with `load_config()`.

---

## Usage Tips

- **Reuse the Pipeline:** For tagging many texts, load the spaCy pipeline once and reuse it across calls.
- **Adjust Batch Settings:** For shorter texts (e.g., tweets), consider reducing `token_normalization` and enable binary tags.
- **Leverage GPU & Multi-processing:** Enable `use_gpu` and adjust `n_processes` to boost performance on large corpora.

---

## Troubleshooting

- **Performance:** If processing is slow, check your GPU and multi-processing settings.
- **spaCy Model:** Ensure that `en_core_web_sm` (or your chosen model) is installed and correctly configured.
- **Configuration Issues:** Validate your configuration by printing the config object to ensure it reflects your intended settings.

---

## References

- Biber, D. (1988). *Variation across Speech and Writing*. Cambridge: Cambridge University Press.
- Grieve, J. (2023). Register variation explains stylometric authorship analysis.
- Additional research and references are detailed in the project documentation.

---

## License

MIT License