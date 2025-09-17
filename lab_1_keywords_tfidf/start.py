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
    frequencies = calculate_frequencies(cleaned_tokens)
    tf_values = calculate_tf(frequencies)
    with open("assets/IDF.json", "r", encoding="utf-8") as file:
        idf = load(file)
    tfidf_values = calculate_tfidf(tf_values, idf)
    with open("assets/corpus_frequencies.json", "r", encoding="utf-8") as file:
        corpus_freqs = load(file)
    result = tfidf_values
    print(get_top_n(result, 10))
    assert result, "Keywords are not extracted"


if __name__ == "__main__":
    main()
