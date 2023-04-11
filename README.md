# Biber MDA

- A pure Python implementation of Biber\'s (1988) Variation across Speech and Writing linguistic tags
- Built upon the spacy library for fast part-of-speech tagging

## Installation

Install the pip package using the GitHub link

```pip install git+https://github.com/davidjurgens/biber-multidimensional-register-analysis```

Once tha package is public...

```pip install bibermda```

## Quickstart Examples

For large text processing make sure to enable GPU with the `use_gpu` flag when loading in the pipeline

### Tagger

**Tag a string**
```python
from bibermda.tagger import load_pipeline, tag_string

pipeline = load_pipeline(use_gpu=False)

# Return a list of tagged values as a dictionary. Best for smaller inputs  
tagged_words = tag_string(pipeline, text)
```

**Tag a large string**
```python
from bibermda.tagger import load_pipeline, tag_large_string

pipeline = load_pipeline(use_gpu=True)

# Output the tags to a compressed TSV file. Best for larger inputs
tag_large_string(pipeline, 
                 text, 
                 out_tsv='large_text.tsv.gz', 
                 show_progess=True)
```

**Tag a large string with parallel processes**

If you have a lot of CPU power, crank up the processes, and it significantly speeds things up!
```python
from bibermda.tagger import tag_string_parallel

tag_string_parallel(pipeline, 
                    text, 
                    out_tsv='large_text.tsv.gz', 
                    token_batch_size=10000, 
                    n_processes=4, 
                    use_gpu=True, 
                    show_progress=False)
```


###  Analyzer

The analyzer will run the tagger and calculate the mean, min, max, range, and standard deviation of tag counts. We use
Biber's default of counting per 1,000 tokens

**Corpus statistics for a text string**
```python
from bibermda.analyzer import calculate_corpus_statistics

statistics_dataframe = calculate_corpus_statistics(text, 
                                                   token_batch_size=10000, 
                                                   token_normalization=1000, 
                                                   use_gpu=True,
                                                   show_progress=False)
```

**Corpus statistics for a large text string**

The main difference is that it uses the parallel tagger to improve the speed. The vast majority of the runtime comes from 
comes from the tagging. 
```python
from bibermda.analyzer import calculate_corpus_statistics_parallel

statistics_dataframe = calculate_corpus_statistics_parallel(text, 
                                                            token_batch_size, 
                                                            token_normalization=1000, 
                                                            n_processes=4, 
                                                            use_gpu=True, 
                                                            show_progress=False)
```

**Corpus statistics for a text directory**

Coming out in the next version

## Generate Low-Dimensional Embeddings

TODO: Release in next version
**PCA**

``` TODO ```

**Factor Analysis**

``` TODO ```

## Use Biber + Grieve Features


## References
- Biber, D. (1988). Variation across Speech and Writing. Cambridge: Cambridge University Press. doi:10.1017/CBO9780511621024
- Nini, A. (2019). The Multi-Dimensional Analysis Tagger. In Berber Sardinha, T. & Veirano Pinto M. (eds), "Multi-Dimensional Analysis: Research Methods and Current Issues", 67-94, London; New York: Bloomsbury Academic. [Download pre-print]
We reference Nini, A. (2019)'s documentation and used their open source software for validation of our library

## LICENSE
MIT License