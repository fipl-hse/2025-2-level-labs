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
    result = calculate_tf \
        (calculate_frequencies(remove_stop_words(clean_and_tokenize(target_text), stop_words)))
    assert result, "Keywords are not extracted"
    print(result)


if __name__ == "__main__":
    main()
