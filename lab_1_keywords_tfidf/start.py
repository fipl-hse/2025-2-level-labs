"""
Frequency-driven keyword extraction starter
"""

# pylint:disable=too-many-locals, unused-argument, unused-variable, invalid-name, duplicate-code
from json import load

from lab_1_keywords_tfidf.main import (
    calculate_chi_values,
    calculate_expected_frequency,
    calculate_frequencies,
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
    tokens = clean_and_tokenize(target_text)
    if not tokens:
        return
    no_stop_words = remove_stop_words(tokens, stop_words)
    if not no_stop_words:
        return
    frequencies = calculate_frequencies(no_stop_words)
    if not frequencies:
        return
    expected_frequencies = calculate_expected_frequency(frequencies, corpus_freqs)
    if not expected_frequencies:
        return
    chi_values = calculate_chi_values(expected_frequencies, frequencies)
    if not chi_values:
        return
    significant_words = extract_significant_words(chi_values, 0.05)
    if not significant_words:
        return
    top_significant = get_top_n(significant_words, 10)
    result = significant_words
    assert result, "Keywords are not extracted"


if __name__ == "__main__":
    main()
