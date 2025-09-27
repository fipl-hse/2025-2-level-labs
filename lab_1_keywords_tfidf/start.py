"""
Frequency-driven keyword extraction starter
"""

# pylint:disable=too-many-locals, unused-argument, unused-variable, invalid-name, duplicate-code
from json import load

from lab_1_keywords_tfidf.main import (
    calculate_frequencies,
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
    
    tokens = clean_and_tokenize(target_text)
    print(f"Tokens: {tokens}")

    filtered_tokens = remove_stop_words(tokens, stop_words)
    print(f"Tokens without stop-words: {filtered_tokens}")

    frequencies = calculate_frequencies(filtered_tokens)
    print(f"Frequencies: {frequencies}")

    top_n = 10
    top_words = get_top_n(frequencies, top_n)
    print(f"Топ-{top_n} самых частых слов:")
    for i, word in enumerate(top_words, 1):
        freq = frequencies[word]
        print(f"{i}. '{word}': {freq} раз")
   
    result = top_words
    assert result, "Keywords are not extracted"


if __name__ == "__main__":
    main()