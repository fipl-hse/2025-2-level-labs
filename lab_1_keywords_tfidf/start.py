"""
Frequency-driven keyword extraction starter
"""

# pylint:disable=too-many-locals, unused-argument, unused-variable, invalid-name, duplicate-code
from json import load
from lab_1_keywords_tfidf.main import (
    clean_and_tokenize,
    remove_stop_words,
    calculate_frequencies,
    get_top_n,
    calculate_tf,
    calculate_tfidf,
    calculate_expected_frequency,
    calculate_chi_values,
    extract_significant_words)


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
    tokens = clean_and_tokenize(target_text)
    no_stop_words = remove_stop_words(tokens, stop_words)
    frequencies = calculate_frequencies(no_stop_words)
    top_frequencies = get_top_n(frequencies, 10)
    tf = calculate_tf(frequencies)
    tfidf = calculate_tfidf(tf, idf)
    top_tfidf = get_top_n(tfidf, 10)
    expected_frequencies = calculate_expected_frequency(frequencies, corpus_freqs)
    chi_values = calculate_chi_values(expected_frequencies, frequencies)
    significant_words = extract_significant_words(chi_values, 0.05)
    top_significant = get_top_n(significant_words, 10)
    result = significant_words
    assert result, "Keywords are not extracted"


if __name__ == "__main__":
    main()
