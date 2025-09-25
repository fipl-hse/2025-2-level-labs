"""
Frequency-driven keyword extraction starter
"""
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
    cleaned_and_tokenized = clean_and_tokenize(target_text)
    if not cleaned_and_tokenized:
        return
    without_stop_words = remove_stop_words(cleaned_and_tokenized, stop_words)
    if not without_stop_words:
        return
    frequencies = calculate_frequencies(without_stop_words)
    if not frequencies:
        return
    expected_frequency = calculate_expected_frequency(frequencies, corpus_freqs)
    if not expected_frequency:
        return
    chi_values = calculate_chi_values(expected_frequency, frequencies)
    if not chi_values:
        return
    significant_words = extract_significant_words(chi_values, alpha=0.05)
    if not significant_words:
        return
    result = get_top_n(significant_words, 10)
    print(result)
    assert result, "Keywords are not extracted"


if __name__ == "__main__":
    main()
