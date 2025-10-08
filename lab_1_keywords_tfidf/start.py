"""
Frequency-driven keyword extraction starter
"""

# pylint:disable=too-many-locals, unused-argument, unused-variable, invalid-name, duplicate-code
from json import load

from main import (
    calculate_frequencies,
    clean_and_tokenize,
    remove_stop_words,
    get_top_n,
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
    if not tokens:
        return
    #print(tokens)
    removed_stop_words = remove_stop_words(tokens,stop_words)
    if not removed_stop_words:
        return
    #print(removed_stop_words)
    frequencies = calculate_frequencies(removed_stop_words)
    if not frequencies:
        return
    #print(frequencies)
    top_n = get_top_n(frequencies, 10)
    #print(top_n)

    result = top_n
    assert result, "Keywords are not extracted"
    
if __name__ == "__main__":
    main()
 