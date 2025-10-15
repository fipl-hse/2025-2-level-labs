"""
Frequency-driven keyword extraction starter
"""

# pylint:disable=too-many-locals, unused-argument, unused-variable, invalid-name, duplicate-code
from json import load

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
    generate_ngrams
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
<<<<<<< HEAD
    if frequencies is not None:
        expected_frequency = calculate_expected_frequency(frequencies, corpus_freqs)
    observed = frequencies
    if expected_frequency is not None and observed is not None:
        chi_values = calculate_chi_values(expected_frequency, observed)
    if chi_values is not None:
        result = extract_significant_words(chi_values, alpha)
    if result is not None:
        print(get_top_n(result, 10))
    n_grams = generate_ngrams(tokens, 3)
    if n_grams is not None:
        frequencies = calculate_frequencies(n_grams)
    if frequencies is not None:
        tf_values = calculate_tf(frequencies)
    with open("assets/IDF.json", "r", encoding="utf-8") as file:
        idf = load(file)
    if tf_values is not None:
        tfidf_values = calculate_tfidf(tf_values, idf)
    with open("assets/corpus_frequencies.json", "r", encoding="utf-8") as file:
        corpus_freqs = load(file)
    if frequencies is not None:
        expected_frequency = calculate_expected_frequency(frequencies, corpus_freqs)
    observed = frequencies
    if expected_frequency is not None and observed is not None:
        chi_values = calculate_chi_values(expected_frequency, observed)
        print(chi_values)
    if chi_values is not None:
        result = extract_significant_words(chi_values, alpha)
    if result is not None:
        print(get_top_n(result, 10))
=======
    tokens = clean_and_tokenize(target_text) or []
    tokens_without_stopwords = remove_stop_words(tokens, stop_words) or []
    print(tokens_without_stopwords)
    frequencies = calculate_frequencies(tokens_without_stopwords) or {}
    term_freq_tf = calculate_tf(frequencies) or {}
    print(term_freq_tf)
    term_freq_tfidf = calculate_tfidf(term_freq_tf, idf) or {}
    print(term_freq_tfidf)
    top_key_words = get_top_n(term_freq_tfidf, 10) or []
    print(', '.join(top_key_words))
    expected = calculate_expected_frequency(frequencies, corpus_freqs) or {}
    chi_values = calculate_chi_values(expected, frequencies) or {}
    significant_words = extract_significant_words(chi_values, alpha=0.001) or {}
    print(significant_words)
    key_words_chi = get_top_n(chi_values, 10) or []
    print(', '.join(key_words_chi))
    result = key_words_chi
>>>>>>> ec42988713a8d3498a174072db58f939816a98ac
    assert result, "Keywords are not extracted"


if __name__ == "__main__":
    main()
