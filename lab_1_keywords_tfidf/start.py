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
    tokens = clean_and_tokenize(target_text) or []
    cleaned_tokens = remove_stop_words(tokens, stop_words) or []
    print(cleaned_tokens)
    freq = calculate_frequencies(cleaned_tokens) or {}
    term_freq = calculate_tf(freq) or {}
    print(term_freq)
    tfidf = calculate_tfidf(term_freq, idf) or {}
    print(tfidf)
    top_key_words = get_top_n(tfidf, 10) or []
    print(top_key_words)
    expected = calculate_expected_frequency(freq, corpus_freqs) or {}
    chi_values = calculate_chi_values(expected, freq) or {}
    significant_words = extract_significant_words(chi_values, alpha=0.001) or {}
    print(significant_words)
    key_words = get_top_n(chi_values, 10) or []
    print(key_words)
    result = key_words
    assert result, "Keywords are not extracted"
        

if __name__ == "__main__":
    main()
