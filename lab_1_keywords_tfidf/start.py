"""
Frequency-driven keyword extraction starter
"""

import json

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
    with open("assets/Дюймовочка.txt", "r", encoding="utf-8") as file:
        target_text = file.read()
        tokens = clean_and_tokenize(target_text)
    with open("assets/stop_words.txt", "r", encoding="utf-8") as file:
        stop_words = file.read().split("\n")
        cleaned_tokens = remove_stop_words(tokens, stop_words)
        freq_dict = calculate_frequencies(cleaned_tokens)
        get_top_words = get_top_n(freq_dict, 5)
    with open("assets/IDF.json", "r", encoding="utf-8") as file:
        idf = json.load(file)
        tf = calculate_tf(freq_dict)
        tfidf = calculate_tfidf(tf, idf)
        top_10_tfidf = get_top_n(tfidf, 10)
    with open("assets/corpus_frequencies.json", "r", encoding="utf-8") as file:
        corpus_freqs = json.load(file)
        expected_freqs = calculate_expected_frequency(freq_dict, corpus_freqs)
        chi_vals = calculate_chi_values(expected_freqs, freq_dict)
        significant_words = extract_significant_words(chi_vals, 0.05)
        top_chi = get_top_n(significant_words, 10) if significant_words else []
        print("Топ-10 ключевых слов по хи-квадрат:", top_chi)
    result = cleaned_tokens
    assert result, "Keywords are not extracted"


if __name__ == "__main__":
    main()