"""
Frequency-driven keyword extraction starter
"""
from json import load

from main import (
    calculate_frequencies,
    calculate_tf,
    clean_and_tokenize,
    get_top_n,
    remove_stop_words,
    calculate_tfidf,
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
    result = get_top_n((calculate_tfidf(calculate_tf(calculate_frequencies(remove_stop_words \
                                                                           (clean_and_tokenize(target_text), stop_words))), idf)), 10)
    print(result)
    assert result, "Keywords are not extracted"


if __name__ == "__main__":
    main()
