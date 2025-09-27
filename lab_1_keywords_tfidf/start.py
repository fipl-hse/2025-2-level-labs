"""
Frequency-driven keyword extraction starter
"""

# pylint:disable=too-many-locals, unused-argument, unused-variable, invalid-name, duplicate-code
from json import load

from lab_1_keywords_tfidf.main import (
    extract_significant_words,
    calculate_chi_values,
    calculate_expected_frequency,
    calculate_frequencies,
    calculate_tf,
    calculate_tfidf,
    clean_and_tokenize,
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
    if not cleaned_tokens:
        return

    removed_stop_words = remove_stop_words(cleaned_tokens, stop_words)
    if not removed_stop_words:
        return

    calculated_frequencies = calculate_frequencies(removed_stop_words)
    if not calculated_frequencies:
        return

    tf = calculate_tf(calculated_frequencies)
    if not tf:
        return

    tfidf = calculate_tfidf(tf, idf)
    if not tfidf:
        return

    top_frequencies = get_top_n(tfidf, 10)
    if not top_frequencies:
        return

    print(top_frequencies)

    expected_frequency = calculate_expected_frequency(calculated_frequencies, corpus_freqs)
    if not expected_frequency:
        return

    chi_values = calculate_chi_values(expected_frequency, calculated_frequencies)
    if not chi_values:
        return

    significant_words = extract_significant_words(chi_values, 0.05)

    top_keywords = get_top_n(significant_words, 10)
    result = top_keywords

    print(result)

    assert result, "Keywords are not extracted"


if __name__ == "__main__":
    main()
