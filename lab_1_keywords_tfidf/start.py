"""
Frequency-driven keyword extraction starter
"""

# pylint:disable=too-many-locals, unused-argument, unused-variable, invalid-name, duplicate-code
from json import load

from lab_1_keywords_tfidf.main import (
    calculate_frequencies,
    calculate_tf,
    calculate_tfidf,
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
    if tokens is None:
        return
    filtered_tokens = remove_stop_words(tokens, stop_words)
    if filtered_tokens is None:
        return
    frequencies = calculate_frequencies(filtered_tokens)
    if frequencies is None:
        return
    tf = calculate_tf(frequencies)
    if tf is None:
        return
    tfidf = calculate_tfidf(tf, idf)
    if tfidf is None:
        return
    result = get_top_n(tfidf, 10)
    if result is None:
        return
    print(result)
    assert result, "Keywords are not extracted"


if __name__ == "__main__":
    main()
