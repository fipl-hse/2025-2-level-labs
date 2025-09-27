"""
Frequency-driven keyword extraction starter
"""

# pylint:disable=too-many-locals, unused-argument, unused-variable, invalid-name, duplicate-code
from json import load

from lab_1_keywords_tfidf.main import calculate_chi_values
from lab_1_keywords_tfidf.main import calculate_expected_frequency
from lab_1_keywords_tfidf.main import calculate_frequencies
from lab_1_keywords_tfidf.main import calculate_tf
from lab_1_keywords_tfidf.main import calculate_tfidf
from lab_1_keywords_tfidf.main import clean_and_tokenize
from lab_1_keywords_tfidf.main import extract_significant_words
from lab_1_keywords_tfidf.main import get_top_n
from lab_1_keywords_tfidf.main import remove_stop_words


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
    clean_words = clean_and_tokenize(target_text) or []
    tokens_list = remove_stop_words(clean_words, stop_words) or []
    freq_dict = calculate_frequencies(tokens_list) or {}
    tf = calculate_tf(freq_dict) or {}
    tfidf = calculate_tfidf(tf, idf) or {}
    expected_freqs = calculate_expected_frequency(freq_dict, corpus_freqs) or {}
    chi_values = calculate_chi_values(expected_freqs, freq_dict) or {}
    significant_words = extract_significant_words(chi_values, 0.001) or {}
    top_tokens = get_top_n(significant_words, 10) or []

    print(top_tokens)
    result = None
#   assert result, "Keywords are not extracted"


if __name__ == "__main__":
    main()
