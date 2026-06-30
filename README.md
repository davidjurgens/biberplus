# Biberplus

[![PyPI version](https://img.shields.io/pypi/v/biberplus.svg?style=flat)](https://pypi.org/project/biberplus/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Biberplus is a pure Python implementation of the linguistic tagging system introduced in Biber (1988). Built upon the spaCy library, it delivers fast part-of-speech tagging along with supplemental features such as a function word tagger, PCA, and factor analysis. These features, inspired by the work of Grieve, Clarke, and colleagues, make Biberplus a powerful tool for analyzing large text corpora.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
  - [Optional extras](#optional-extras)
- [Quickstart Guide](#quickstart-guide)
  - [Biber Tagger](#1-biber-tagger)
  - [Tagging a Corpus](#2-tagging-a-corpus)
  - [Function Words Tagger](#3-function-words-tagger)
  - [Word-Level Tagging](#4-word-level-tagging)
  - [Text Embeddings](#5-text-embeddings)
  - [Dimension Reduction](#6-dimension-reduction)
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

```bash
pip install biberplus
```

Requires:

- Python 3.10+
- pandas, NumPy, spaCy, tqdm, PyYAML, factor_analyzer.
- a spaCy English language model

After installation, download a spaCy model with:

```bash
python -m spacy download en_core_web_sm
```

### Optional extras

For PCA/factor analysis plotting functions:

```bash
pip install biberplus[plots]
```

For development (ruff, flake8):

```bash
pip install biberplus[dev]
```

Then run linting with:

```bash
ruff check .
flake8 .
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

**Understanding the frequencies output**

`calculate_tag_frequencies` slices the text into windows of `token_normalization`
tokens (default 100), counts each tag per window, and then reports descriptive
statistics across those windows. The returned DataFrame has one row per tag with
these columns:

| column    | meaning                                                        |
|-----------|----------------------------------------------------------------|
| `tag`     | the feature name (e.g. `NN`, `PASS`, `BIN_PASS`, `AWL`, `TTR`) |
| `mean`    | average per-window count (per `token_normalization` tokens)    |
| `min_val` | smallest per-window count                                      |
| `max_val` | largest per-window count                                       |
| `range`   | `max_val - min_val`                                            |
| `std`     | standard deviation across windows                              |

Special rows: `AWL` is the average word length and `TTR` is the type-token ratio
(over the first 400 tokens). When `binary_tags` is enabled, each Biber tag also
gets a `BIN_<tag>` row whose `mean` is the **fraction of windows** in which the
feature appeared at least once (so `std` equals `sqrt(mean * (1 - mean))`).

### 2. Tagging a Corpus

When tagging many documents, build the config and spaCy pipeline **once** and
reuse them in a loop. Reloading the pipeline per document is the most common
performance pitfall.

```python
from biberplus.tagger import load_config, load_pipeline, calculate_tag_frequencies

corpus = ["First document ...", "Second document ...", "Third document ..."]

config = load_config()
config.update({'function_words': False})
pipeline = load_pipeline(config)  # load the model only once

per_doc_frequencies = [
    calculate_tag_frequencies(text, pipeline, config) for text in corpus
]
```

**Parallel tagging of a single large document (GPU / multi-processing):**

```python
from biberplus.tagger import load_config, load_pipeline, calculate_tag_frequencies

config = load_config()
config.update({'use_gpu': True, 'n_processes': 4, 'function_words': False})
pipeline = load_pipeline(config)
frequencies_df = calculate_tag_frequencies("Your very long text ...", pipeline, config)
print(frequencies_df)
```

> **Windows / macOS note:** these platforms start worker processes with the
> `spawn` method, which re-imports your script. Wrap the entry point in a
> `if __name__ == "__main__":` guard so multi-processing does not re-execute your
> top-level code in every worker:
>
> ```python
> if __name__ == "__main__":
>     main()
> ```

### 3. Function Words Tagger

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
    'function_words_list': custom_fw
})
frequencies_df = calculate_tag_frequencies("Your sample text goes here", config=config)
print(frequencies_df)
```

### 4. Word-Level Tagging

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

### 5. Text Embeddings

Generate an embedding from the textual data. The result is grouped by feature
type (`biber`, and optionally `binary` and `function_words`), and each group
carries both the `values` vector and the aligned feature `names`, so you always
know which feature each number corresponds to:

```python
from biberplus.tagger import load_config
from biberplus.reducer import encode_text

config = load_config()
config.update({'binary_tags': True, 'function_words': True})
encoding = encode_text(config, "Your sample text goes here")

# The 'biber' group uses all five statistics per tag.
biber = encoding['biber']
print(biber['values'][:5])  # e.g. [0.0, 0.0, 0.0, 0.0, 0.0]
print(biber['names'][:5])   # e.g. ['QUAN_mean', 'QUAN_min_val', 'QUAN_max_val', 'QUAN_range', 'QUAN_std']

# The 'binary' group only carries mean/std per tag (presence indicators).
binary = encoding['binary']
print(binary['names'][:4])  # e.g. ['BIN_QUAN_mean', 'BIN_QUAN_std', 'BIN_QUPR_mean', 'BIN_QUPR_std']

# The 'function_words' group has one block per function word.
function_words = encoding['function_words']
print(function_words['names'][:5])

# Names and values are positionally aligned within every group:
named_features = dict(zip(biber['names'], biber['values']))
```

If you already have a frequencies DataFrame from `calculate_tag_frequencies`
(which includes a human-readable `tag` column), encode it directly without
re-tagging:

```python
from biberplus.tagger import load_config, calculate_tag_frequencies
from biberplus.reducer import encode_frequencies

config = load_config()
config.update({'binary_tags': True})
frequencies_df = calculate_tag_frequencies("Your sample text goes here", config=config)

encoding = encode_frequencies(frequencies_df, config)
print(encoding['biber']['names'][:5])
print(encoding['biber']['values'][:5])
```

### 6. Dimension Reduction

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

---

## Funding

This research is supported in part by the Office of the Director of National Intelligence (ODNI), Intelligence Advanced Research Projects Activity (IARPA), via the HIATUS Program contract #2022-22072200006. The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies, either expressed or implied, of ODNI, IARPA, or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for governmental purposes notwithstanding any copyright annotation therein.
