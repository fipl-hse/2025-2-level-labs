"""
Frequency-driven keyword extraction starter
"""

# pylint:disable=too-many-locals, unused-argument, unused-variable, invalid-name, duplicate-code
from json import load
#lab_1_keywords_tfidf.
from main import (
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
    cleaned_tokens = None
    frequencies = None
    tf_values = None
    alpha = 0.001
    tfidf_values = None
    expected_frequency = None
    chi_values = None
    result = None
    with open("assets/Дюймовочка.txt", "r", encoding="utf-8") as file:
        target_text = file.read()
    with open("assets/stop_words.txt", "r", encoding="utf-8") as file:
        stop_words = file.read().split("\n")
    with open("assets/IDF.json", "r", encoding="utf-8") as file:
        idf = load(file)
    with open("assets/corpus_frequencies.json", "r", encoding="utf-8") as file:
        corpus_freqs = load(file)
    tokens = clean_and_tokenize(target_text)
    if tokens is not None:
        cleaned_tokens = remove_stop_words(tokens, stop_words)
    if cleaned_tokens is not None:
        frequencies = calculate_frequencies(cleaned_tokens)
    if frequencies is not None:
        tf_values = calculate_tf(frequencies)
    if tf_values is not None:
        tfidf_values = calculate_tfidf(tf_values, idf)
    #print(get_top_n(tfidf_values, 10))
    if frequencies is not None:
        expected_frequency = calculate_expected_frequency(frequencies, corpus_freqs)
    observed = frequencies
    if expected_frequency is not None and observed is not None:
        chi_values = calculate_chi_values(expected_frequency, observed)
    if chi_values is not None:
        result = extract_significant_words(chi_values, alpha)
    if result is not None:
        print(get_top_n(result, 10))
    assert result, "Keywords are not extracted"


if __name__ == "__main__":
    main()
