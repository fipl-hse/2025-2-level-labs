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
    tokens = clean_and_tokenize(target_text) or []
    with open("assets/stop_words.txt", "r", encoding="utf-8") as file:
        stop_words = file.read().split("\n")
    cleaned_tokens = remove_stop_words(tokens, stop_words) or []
    frequencies = calculate_frequencies(cleaned_tokens) or {}
    top_n = get_top_n(frequencies, 5) or []
    tf = calculate_tf(frequencies) or {}
    with open("assets/IDF.json", "r", encoding="utf-8") as file:
        idf = load(file)
    tf_idf = calculate_tfidf(tf, idf) or {}
    top_idf = get_top_n(tf_idf, 10) or []
    with open("assets/corpus_frequencies.json", "r", encoding="utf-8") as file:
        corpus_freqs = load(file)
    exp_freqs_dict = calculate_expected_frequency(frequencies, corpus_freqs) or {}
    chi_values = calculate_chi_values(corpus_freqs_dict, frequencies) or {}
    sign_chi_values = extract_significant_words(chi_values, 0.05) or {}
    top_sign_chi_values = get_top_n(sign_chi_values, 10) or []
    print(top_sign_chi_values)
    result = top_sign_chi_values
    assert result, "Keywords are not extracted"


if __name__ == "__main__":
    main()
