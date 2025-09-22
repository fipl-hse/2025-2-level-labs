"""
Frequency-driven keyword extraction starter
"""

# pylint:disable=too-many-locals, unused-argument, unused-variable, invalid-name, duplicate-code
from main import (
    calculate_chi_values,
    calculate_expected_frequency,
    calculate_frequencies,
    calculate_tf,
    calculate_tfidf,
    clean_and_tokenize,
    extract_significant_words,
    get_top_n,
    remove_stop_words
    ) 


from json import load


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
    cleaned_tokens = clean_and_tokenize(target_text) or []
    removed_stop_words = remove_stop_words(cleaned_tokens, stop_words) or []
    calculated_frequencies = calculate_frequencies(removed_stop_words) or {}
    expected_frequency = calculate_expected_frequency(calculate_frequencies, corpus_freqs) or {}
    chi_values = calculate_chi_values(expected_frequency, calculated_frequencies) or {}
    significant_words = extract_significant_words(chi_values, 0.005) or {}
    top_words = get_top_n(significant_words, 6) or []
    result = None
    assert result, "Keywords are not extracted"


if __name__ == "__main__":
    main()
