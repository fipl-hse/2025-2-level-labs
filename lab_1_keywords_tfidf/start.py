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
    result = None
    assert result, "Keywords are not extracted"
    unfilt_tokens = clean_and_tokenize(target_text)
    print(unfilt_tokens)
    tokens = remove_stop_words(unfilt_tokens, stop_words)
    print (tokens)
    frequencies = calculate_frequencies(tokens)
    print (frequencies)
    term_freq = calculate_tf(frequencies)
    print (term_freq)
    tfidf_dict = calculate_tfidf(term_freq, idf)
    print (tfidf_dict)
    print (get_top_n(tfidf_dict, 10))
    result = tfidf_dict

if __name__ == "__main__":
    main()
