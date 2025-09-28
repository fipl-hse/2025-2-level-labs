"""
Frequency-driven keyword extraction starter
"""

# pylint:disable=too-many-locals, unused-argument, unused-variable, invalid-name, duplicate-code
import os
from json import load

try:
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
except ImportError:
    from .main import (
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
    # Получаем путь к директории текущего файла
    current_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.join(current_dir, "assets")
    
    with open(os.path.join(assets_dir, "Дюймовочка.txt"), "r", encoding="utf-8") as file:
        target_text = file.read()
    with open(os.path.join(assets_dir, "stop_words.txt"), "r", encoding="utf-8") as file:
        stop_words = file.read().split("\n")
    with open(os.path.join(assets_dir, "IDF.json"), "r", encoding="utf-8") as file:
        idf = load(file)
    with open(os.path.join(assets_dir, "corpus_frequencies.json"), "r", encoding="utf-8") as file:
        corpus_freqs = load(file)
    
    tokenized_text = clean_and_tokenize(target_text)
    if not tokenized_text:
        return

    text_without_stopwords = remove_stop_words(tokenized_text, stop_words)
    if not text_without_stopwords:
        return

    frequ_dict = calculate_frequencies(text_without_stopwords)
    if not frequ_dict:
        return

    tf_dict = calculate_tf(frequ_dict)
    if not tf_dict:
        return

    tfidf_dict = calculate_tfidf(tf_dict, idf)
    if not tfidf_dict:
        return

    expected_freq = calculate_expected_frequency(frequ_dict, corpus_freqs)
    if not expected_freq:
        return

    chi_values = calculate_chi_values(expected_freq, frequ_dict) or {}

    significant_words = extract_significant_words(chi_values, 0.01) or {}
    top_significant_words = get_top_n(significant_words, 10)
    result = top_significant_words

    print(result)
    assert result, "Keywords are not extracted"



main()