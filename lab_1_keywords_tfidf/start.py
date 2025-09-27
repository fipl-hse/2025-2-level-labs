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
    wo_stop_words = remove_stop_words(tokens, stop_words) or []
    frequencies = calculate_frequencies(wo_stop_words) or {}
    term_frequencies = calculate_tf(frequencies) or {}
    tf_idf = calculate_tfidf(term_frequencies, idf) or {}
    expected = calculate_expected_frequency(frequencies, corpus_freqs) or {}
    chi_values = calculate_chi_values(expected, frequencies) or {}
    significant_words = extract_significant_words(chi_values, 0.001) or {}
    top_n_3 = get_top_n(significant_words, 10) or []
    print(top_n_3)
    result = None
    #assert result, "Keywords are not extracted"




if __name__ == "__main__":
    main()
