"""
Frequency-driven keyword extraction starter
"""

# pylint:disable=too-many-locals, unused-argument, unused-variable, invalid-name, duplicate-code

from json import load

try:
    from main import (
        calculate_chi_values,
        calculate_expected_frequency,
        calculate_frequencies,
        calculate_tf,
        calculate_tfidf,
        clean_and_tokenize,
        extract_significant_words,
        get_top_n,
        remove_stop_words,
    )
except ImportError:
    from .main import (
        calculate_chi_values,
        calculate_expected_frequency,
        calculate_frequencies,
        calculate_tf,
        calculate_tfidf,
        clean_and_tokenize,
        extract_significant_words,
        get_top_n,
        remove_stop_words,
    )


def main() -> None:
    """
    Launches an implementation.
    """

    with open("assets/Дюймовочка.txt", "r", encoding="utf-8") as file:
        target_text = file.read()
    with open("assets/stop_words.txt", "r", encoding="utf-8") as file:
        stop_words = file.read().split("\n")
    with open("assets/IDF.json", "r", encoding="utf-8") as file:
        idf = load(file)
    with open("corpus_frequencies.json", "r", encoding="utf-8") as file:
        corpus_freqs = load(file)

    tokenized_text = clean_and_tokenize(target_text)
    if not tokenized_text:
        return
    #print(tokenized_text)

    text_without_stopwords = remove_stop_words(tokenized_text, stop_words)
    if not text_without_stopwords:
        return
    #print(text_without_stopwords)

    frequ_dict = calculate_frequencies(text_without_stopwords)
    if not frequ_dict:
        return
    #print(frequ_dict)

    tf_dict = calculate_tf(frequ_dict)
    if not tf_dict:
        return
    #print(tf_dict)

    tfidf_dict = calculate_tfidf(tf_dict, idf)
    if not tfidf_dict:
        return
    #print(tfidf_dict)

    expected_freq = calculate_expected_frequency(frequ_dict, corpus_freqs)
    if not expected_freq:
        return
    #print(expected_freq)

    chi_values = calculate_chi_values(expected_freq, frequ_dict) or {}

    #print(chi_values)

    significant_words = extract_significant_words(chi_values, 0.01) or {}
    top_significant_words = get_top_n(significant_words, 10)
    result = top_significant_words

    #print(significant_words)
    #print(top_significant_words)

    print(result)
    assert result, "Keywords are not extracted"

if __name__ == "__main__":
    main()
