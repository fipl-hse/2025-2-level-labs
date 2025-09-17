"""
Frequency-driven keyword extraction starter
"""

# pylint:disable=too-many-locals, unused-argument, unused-variable, invalid-name, duplicate-code
from json import load

from main import (clean_and_tokenize, remove_stop_words, calculate_frequencies, get_top_n, calculate_tf, calculate_tfidf,calculate_expected_frequency, calculate_chi_values)

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
    tokens = clean_and_tokenize(target_text)
    filteted_tokens = remove_stop_words(tokens, stop_words)
    frequencies = calculate_frequencies(filteted_tokens)
    result = get_top_n(frequencies, 10)
    print(result)
    assert result, "Keywords are not extracted"

if __name__ == "__main__":
    main()
