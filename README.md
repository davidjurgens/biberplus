# Biber MDA

- A pure Python implementation of Biber\'s (1988) Variation across Speech and Writing linguistic tags
- Includes supplemental features from the work of Grieve, Clarke, et al.
- Built-in function word tagger, PCA, and factor analysis
- Built upon the spacy library for fast part-of-speech tagging

## Installation

Install the pip package using the GitHub link

```pip install git+https://github.com/davidjurgens/biberplus```

Once tha package is public...

```pip install biberplus```

## Quickstart Examples

For large text processing make sure to enable GPU with the `use_gpu` flag in the configuration. To use multiple
CPUs, set `n_processes` in the configuration.

For tagging millions of smaller texts (e.g. tweets or Reddit comments) check out the examples in `corpora_tagging`

### Biber Tagger

The default return value is a DataFrame containing the frequencies per linguistic tag for every N tokens.
The tagger will calculate the mean, min, max, range, and standard deviation of tag counts. We use Biber's default of
counting per 1,000 tokens. For social media text, we recommend you reduce the value to 100 and use the binary features

**Tag a string with the default configuration**

```python
from biberplus.tagger import calculate_tag_frequencies

frequencies_df = calculate_tag_frequencies(text)
```

**Tag a large corpus with multi-processing**

Use the GPU and multi-processing to speed things up. Modify from the default configuration.

```python
from biberplus.tagger import load_config, load_pipeline, calculate_tag_frequencies

config = load_config()
config.update({'use_gpu': True, 'n_processes': 4, 'function_words': False})
pipeline = load_pipeline(config)
frequencies_df = calculate_tag_frequencies(text, pipeline, config)
```

**Return the tagged text rather than the calculated frequencies**

If you are calling `tag_text` on many strings, load in the Spacy pipeline once with `load_pipeline`

```python
from biberplus.tagger import tag_text, load_config, load_pipeline

# If you're calling tag_text repeatedly, load in the pipeline only once to improve performance
config = load_config()
pipeline = load_pipeline(config)
tagged_words = tag_text(text, pipeline=pipeline)

# Otherwise call tax_text directly
tagged_words = tag_text(text)
```

---

### Function Words Tagger

**Using the default list provided**

The default behavior uses 100 of the most frequent function words. You may pass in your own list of functions words as
as a variable. The default list of function words can be found at `biberplus/tagger/constants/function_words.txt`

```python
from biberplus.tagger import load_config, calculate_tag_frequencies

config = load_config()
config.update({'use_gpu': True, 'biber': False, 'function_words': True})
frequencies_df = calculate_tag_frequencies(text)
```

**Using your own list**

```python
from biberplus.tagger import load_config, calculate_tag_frequencies

function_words = [...]
config = load_config()
config.update({'function_words': True, 'biber': False, 'grieve_clarke': False,
               'function_words_list': function_words})
frequencies_df = calculate_tag_frequencies(text, function_words)
```

---

## Encode Text To Embeddings

The 67x5 Biber frequencies DataFrame is flattened into a vector of length 335
Similarly the function words and additional Grieve/Clarke tag frequencies are f
The output length of the vector is dependent on the input options

**Biber + Function Words example embedding**

```python
from biberplus.tagger import load_config
from biberplus.reducer import encode_text

config = load_config()
embedding = encode_text(text, config)
```

---

## Dimension Reduction

### PCA

**Biber + Function Words PCA**

```python
from biberplus.tagger import load_config, load_pipeline, calculate_tag_frequencies
from biberplus.reducer import tags_pca

config = load_config()
config.update({'use_gpu': True, 'biber': True, 'function_words': True})
pipeline = load_pipeline(config)
frequencies_df = calculate_tag_frequencies(text, pipeline, config)

pca_df, explained_variance = tags_pca(frequencies_df, components=2)
```

### Factor Analysis

**Biber + Function Words PCA**


---

## References

- Biber, D. (1988). Variation across Speech and Writing. Cambridge: Cambridge University Press. doi:
  10.1017/CBO9780511621024
- Nini, A. (2019). The Multi-Dimensional Analysis Tagger. In Berber Sardinha, T. & Veirano Pinto M. (eds), "
  Multi-Dimensional Analysis: Research Methods and Current Issues", 67-94, London; New York: Bloomsbury
  Academic. 
  We reference Nini, A. (2019)'s documentation and used their open source software for validation of our library
- Grieve, J. (2023). Register variation explains stylometric authorship analysis. Corpus Linguistics and Linguistic
  Theory, 19(1), 47-77. https://doi.org/10.1515/cllt-2022-0040
- Dimensions of Abusive Language on Twitter (Clarke & Grieve, ALW 2017)
- Grieve, J., Biber, D., Friginal, E., Nekrasova, T. (2010). Variation Among Blogs: A Multi-dimensional Analysis. In:
  Mehler, A., Sharoff, S., Santini, M. (eds) Genres on the Web. Text, Speech and Language Technology, vol 42. Springer,
  Dordrecht. https://doi.org/10.1007/978-90-481-9178-9_14

## LICENSE

MIT License