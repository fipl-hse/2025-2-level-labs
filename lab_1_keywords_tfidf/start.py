"""
Frequency-driven keyword extraction starter
"""
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
    if not clean_and_tokenize(target_text):
        return None
    cleaned_and_tokenized = clean_and_tokenize(target_text)
    if not remove_stop_words(cleaned_and_tokenized, stop_words):
        return None
    without_stop_words = remove_stop_words(cleaned_and_tokenized, stop_words)
    if not calculate_frequencies(without_stop_words):
        return None
    frequencies = calculate_frequencies(without_stop_words)
    if not calculate_expected_frequency(frequencies, corpus_freqs):
        return None
    expected_frequency = calculate_expected_frequency(frequencies, corpus_freqs)
    if not calculate_chi_values(expected_frequency, frequencies):
        return None
    chi_values = calculate_chi_values(expected_frequency, frequencies)
    if not extract_significant_words(chi_values, alpha=0.05):
        return None
    significant_words = extract_significant_words(chi_values, alpha=0.05)
    if not get_top_n(chi_values, 10):
        return None
    result = get_top_n(significant_words, 10)
    print(result)
    assert result, "Keywords are not extracted"


if __name__ == "__main__":
    main()
