from bibermda.tagger import calculate_tag_frequencies


def encode_text(config, text):
    frequencies_df = calculate_tag_frequencies(text, config=config)
    return frequencies_df.to_numpy().flatten()
