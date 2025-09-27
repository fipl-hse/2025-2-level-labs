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

    cleaned_text=clean_and_tokenize(target_text)
    if cleaned_text is None:
        return
    without_stop_words=remove_stop_words(cleaned_text, stop_words)
    if without_stop_words is None:
        return
    calculated_frequencies = calculate_frequencies(without_stop_words)
    if calculated_frequencies is None:
        return
    calculated_tf=calculate_tf(calculated_frequencies)
    if calculated_tf is None:
        return
    calculated_tfidf=calculate_tfidf(calculated_tf, idf)
    if calculated_tfidf is None:
        return
    expected_frequency=calculate_expected_frequency(calculated_frequencies, corpus_freqs)
    if expected_frequency is None:
        return
    chi_values=calculate_chi_values(expected_frequency, calculated_frequencies)
    if chi_values is None:
        return
    significant_words=extract_significant_words(chi_values, 0.05)
    if significant_words is None:
        return
    get_top_words=get_top_n(significant_words, 10)
    if get_top_words is None:
        return

    result=get_top_words
    print(result)
    assert result, "Keywords are not extracted"

if __name__ == "__main__":
    main()
