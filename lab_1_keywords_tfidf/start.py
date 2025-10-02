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

    cleaned_tokens = clean_and_tokenize(target_text)
    if cleaned_tokens is None:
        print("Error: No tokens after cleaning")
        return

    removed_stop_words = remove_stop_words(cleaned_tokens, stop_words)
    if removed_stop_words is None:
        print("Error: No words after stop words removal")
        return

    calculated_frequencies = calculate_frequencies(removed_stop_words)
    if calculated_frequencies is None:
        print("Error: Could not calculate frequencies")
        return

    expected_frequency = calculate_expected_frequency(calculated_frequencies, corpus_freqs)
    if expected_frequency is None:
        print("Error: Could not calculate expected frequency")
        return

    chi_values = calculate_chi_values(expected_frequency, calculated_frequencies)
    if chi_values is None:
        print("Error: Could not calculate chi values")
        return

    significant_words = extract_significant_words(chi_values, 0.05)
    if significant_words is None:
        print("Error: No significant words found")
        return

    result = get_top_n(significant_words, 10)
    if result is None:
        print("Error: Could not get top words")
        return

    assert result, "Keywords are not extracted"
    print("Keywords:", result)

if __name__ == "__main__":
    main()
