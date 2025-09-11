"""
Frequency-driven keyword extraction starter
"""

# pylint:disable=too-many-locals, unused-argument, unused-variable, invalid-name, duplicate-code
import os
from json import load
from .main import (
    clean_and_tokenize,
    remove_stop_words,
    calculate_frequencies,
    get_top_n,
    calculate_tf,
    calculate_tfidf,
    calculate_expected_frequency,
    calculate_chi_values,
    extract_significant_words
)


BASE_DIR = os.path.dirname(__file__)
ASSETS_DIR = os.path.join(BASE_DIR, "assets")


def main() -> None:
    """
    Launches an implementation.
    """
    with open(f"{ASSETS_DIR}/Дюймовочка.txt", "r", encoding="utf-8") as file:
        target_text = file.read()
    with open(f"{ASSETS_DIR}/stop_words.txt", "r", encoding="utf-8") as file:
        stop_words = file.read().split("\n")
    with open(f"{ASSETS_DIR}/IDF.json", "r", encoding="utf-8") as file:
        idf = load(file)
    with open(f"{ASSETS_DIR}/corpus_frequencies.json", "r", encoding="utf-8") as file:
        corpus_freqs = load(file)


    cleaned_tokens = clean_and_tokenize(target_text) #1
    # print(cleaned_tokens)

    removed_stop_words = remove_stop_words(cleaned_tokens, stop_words) #2
    # print(removed_stop_words)

    calculated_frequencies = calculate_frequencies(removed_stop_words) #3
    # print(calculated_frequencies)

    top_words = get_top_n(calculated_frequencies, 5) #4
    # print(top_words)

    term_frequency = calculate_tf(calculated_frequencies) #5
    # print(term_frequency)

    tf_idf = calculate_tfidf(term_frequency, idf) #6
    # print(tf_tdf)

    top_words = get_top_n(tf_idf, 10) #7
    # print(top_words)

    expected_frequency = calculate_expected_frequency(calculated_frequencies, corpus_freqs) #8
    # print(expected_frequency)

    chi_values = calculate_chi_values(expected_frequency, calculated_frequencies) #9
    # print(chi_values)

    significant_words = extract_significant_words(chi_values, 0.05) #10
    # print(significant_words)

    top_words = get_top_n(significant_words, 10) #11
    print(top_words)

    # result = None
    # assert result, "Keywords are not extracted"


if __name__ == "__main__":
    main()
