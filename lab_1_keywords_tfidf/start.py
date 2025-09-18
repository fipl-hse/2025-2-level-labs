"""
Frequency-driven keyword extraction starter
"""

# pylint:disable=too-many-locals, unused-argument, unused-variable, invalid-name, duplicate-code
from main import clean_and_tokenize
from main import remove_stop_words
from main import calculate_frequencies
from main import get_top_n
from main import calculate_tf
from main import calculate_tfidf
from main import calculate_expected_frequency
from main import calculate_chi_values
from main import extract_significant_words

from json import load


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
    #result = None
    #assert result, "Keywords are not extracted"

    tokens = clean_and_tokenize(target_text)
    print(f"Tokens (first 20): {tokens[:20]}")

if __name__ == "__main__":
    main()

