"""
Frequency-driven keyword extraction starter
"""

# pylint:disable=too-many-locals, unused-argument, unused-variable
# pylint:disable=invalid-name, duplicate-code

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
    with open("Дюймовочка.txt", "r", encoding="utf-8") as file:
        target_text = file.read()
    with open("stop_words.txt", "r", encoding="utf-8") as file:
        stop_words = file.read().split("\n")
    with open("corpus_frequencies.json", "r", encoding="utf-8") as file:
        corpus_freqs = load(file)

    cleaned_tokens = clean_and_tokenize(target_text)
    if not cleaned_tokens:
        return

    removed_stop_words = remove_stop_words(cleaned_tokens, stop_words)
    if not removed_stop_words:
        return

    calculated_frequencies = calculate_frequencies(removed_stop_words)
    if not calculated_frequencies:
        return

    expected_frequency = calculate_expected_frequency(
        calculated_frequencies,
        corpus_freqs
    )
    if not expected_frequency:
        return

    chi_values = calculate_chi_values(
        expected_frequency,
        calculated_frequencies
    )
    if not chi_values:
        return

    significant_words = extract_significant_words(chi_values, 0.05)
    if not significant_words:
        return

    top_words = get_top_n(significant_words, 10)

    if top_words is None:
        raise ValueError("No keywords extracted")

    assert len(top_words) == 10, f"Expected 10 keywords, got {len(top_words)}"

    print(top_words)


if __name__ == "__main__":
    main()
