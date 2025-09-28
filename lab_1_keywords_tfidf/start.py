"""
Frequency-driven keyword extraction starter
"""

# pylint:disable=too-many-locals, unused-argument, unused-variable, invalid-name, duplicate-code
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

    cleaned_text=clean_and_tokenize(target_text) or []
    without_stop_words=remove_stop_words(cleaned_text, stop_words)
    calculated_frequencies = calculate_frequencies(without_stop_words)
    expected_frequency=calculate_expected_frequency(calculated_frequencies, corpus_freqs)
    chi_values=calculate_chi_values(expected_frequency, calculated_frequencies)
    significant_words=extract_significant_words(chi_values, 0.05)
    get_top_words=get_top_n(significant_words, 10)

    result=get_top_words
    if result:
        print(result)
    assert result, "Keywords are not extracted"

if __name__ == "__main__":
    main()
