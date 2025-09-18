"""
Frequency-driven keyword extraction starter
"""

# pylint:disable=too-many-locals, unused-argument, unused-variable, invalid-name, duplicate-code
from json import load

from main import (
    calculate_frequencies,
    clean_and_tokenize,
    remove_stop_words,
)


def main() -> None:
    """
    Launches an implementation.
    """
    with open("lab_1_keywords_tfidf/assets/Дюймовочка.txt", "r", encoding="utf-8") as file:
        target_text = file.read()
    with open("lab_1_keywords_tfidf/assets/stop_words.txt", "r", encoding="utf-8") as file:
        stop_words = file.read().split("\n")
    with open("lab_1_keywords_tfidf/assets/IDF.json", "r", encoding="utf-8") as file:
        idf = load(file)
    with open("lab_1_keywords_tfidf/assets/corpus_frequencies.json", "r", encoding="utf-8") as file:
        corpus_freqs = load(file)

    print(clean_and_tokenize.__name__)
    cleaned_tokens = clean_and_tokenize(target_text)
    if not cleaned_tokens:
        return

    removed_stop_words = remove_stop_words(cleaned_tokens, stop_words)
    if not removed_stop_words:
        return

    calculated_frequencies = calculate_frequencies(removed_stop_words)
    if not calculated_frequencies:
        return

    result = None
    assert result, "Keywords are not extracted"


if __name__ == "__main__":
    main()
