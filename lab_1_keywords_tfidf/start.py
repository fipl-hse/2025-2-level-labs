"""
Lab 1 start file
"""

from json import load
from lab_1_keywords_tfidf.main import (
    clean_and_tokenize,
    remove_stop_words,
    calculate_frequencies,
    get_top_n,
    calculate_tf,
    calculate_tfidf,
    calculate_expected_frequency
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

    tokens = clean_and_tokenize(target_text)
    if tokens is None:
        print("Error: Could not tokenize text")
        return

    tokens = clean_and_tokenize(target_text)
    filtered_tokens = remove_stop_words(tokens, stop_words)
    frequencies = calculate_frequencies(filtered_tokens)
    freq_keywords = get_top_n(frequencies, 10)
    print("Frequency keywords:", freq_keywords)

    tf_scores = calculate_tf(frequencies)
    tfidf_scores = calculate_tfidf(tf_scores, idf)
    tfidf_keywords = get_top_n(tfidf_scores, 10)
    print("TF-IDF keywords:", tfidf_keywords)

    expected_freqs = calculate_expected_frequency(frequencies, corpus_freqs)
    expected_keywords = get_top_n(expected_freqs, 10)
    print("Expected frequency keywords:", expected_keywords)

    result = freq_keywords
    assert result, "Keywords are not extracted"


if __name__ == "__main__":
    main()
