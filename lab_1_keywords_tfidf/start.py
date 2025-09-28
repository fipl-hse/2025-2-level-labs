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

    tokens = clean_and_tokenize(target_text) or []
    tokens_wo_stop_words = remove_stop_words(tokens, stop_words) or []
    frequencies = calculate_frequencies(tokens_wo_stop_words) or {}
    expected_freq = calculate_expected_frequency(frequencies, corpus_freqs) or {}
    chi_values = calculate_chi_values(expected_freq, frequencies) or {}
    significant_words = extract_significant_words(chi_values, 0.01) or {}
    result = get_top_n(significant_words, 10)
    print(result)
    assert result, "Keywords are not extracted"





if __name__ == "__main__":
    main()
