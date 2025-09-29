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
    with open("assets/corpus_frequencies.json", "r", encoding="utf-8") as file:
        corpus_freqs = load(file)

    result = None
    cleaned_tokens = None
    removed_stop_words = None
    calculated_frequencies = None
    expected_frequency = None
    chi_values = None
    significant_words = None

    cleaned_tokens = clean_and_tokenize(target_text)
    if cleaned_tokens:
        removed_stop_words = remove_stop_words(cleaned_tokens, stop_words)
    if cleaned_tokens and removed_stop_words:
        calculated_frequencies = calculate_frequencies(removed_stop_words)
    conditions = (cleaned_tokens, removed_stop_words, calculated_frequencies)
    if all(conditions):
        expected_frequency = calculate_expected_frequency(
            calculated_frequencies, corpus_freqs
        )
    conditions_2 = conditions + (expected_frequency, )
    if all(conditions_2):
        chi_values = calculate_chi_values(expected_frequency, calculated_frequencies)
    conditions_3 = conditions_2 + (chi_values, )
    if all(conditions_3):
        significant_words = extract_significant_words(chi_values, 0.05)
    conditions_4 = conditions_3 + (significant_words, )
    if all(conditions_4):
        result = get_top_n(significant_words, 10)

    if not result:
        print("Error: Keywords are not extracted")
        return

    assert result, "Keywords are not extracted"
    print("Keywords:", result)

if __name__ == "__main__":
    main()
