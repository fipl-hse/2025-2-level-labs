"""
Frequency-driven keyword extraction starter
"""

# pylint:disable=too-many-locals, unused-argument, unused-variable, invalid-name, duplicate-code
from json import load

from lab_1_keywords_tfidf.main import (
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
    with open("assets/corpus_frequencies.json", "r", encoding="utf-8") as file:
        corpus_freqs = load(file)
    if target_text:
        tokenized_text = clean_and_tokenize(target_text)
    if tokenized_text:
        text_without_stopwords = remove_stop_words(tokenized_text, stop_words)
    if text_without_stopwords:
        frequ_dict = calculate_frequencies(text_without_stopwords)
    if frequ_dict:
        tf_dict = calculate_tf(frequ_dict)
    if tf_dict:
        tfidf_dict = calculate_tfidf(tf_dict, idf)
    if tf_dict and tfidf_dict:
        expected_freq = calculate_expected_frequency(frequ_dict, corpus_freqs)
    if expected_freq and tfidf_dict:
        chi_values = calculate_chi_values(expected_freq, tfidf_dict)
    if chi_values:
        significant_words = extract_significant_words(chi_values, 0.01)
    if significant_words:
        top_significant_words = get_top_n(significant_words, 10)
    result = top_significant_words

    print(result)
    assert result, "Keywords are not extracted"


if __name__ == "__main__":
    main()
