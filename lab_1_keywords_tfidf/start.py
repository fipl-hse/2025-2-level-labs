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
    tokens = clean_and_tokenize(target_text)
    print(tokens)
    if tokens is None:
        return
    with open("assets/stop_words.txt", "r", encoding="utf-8") as file:
        stop_words = file.read().split("\n")
    cleaned_tokens = remove_stop_words(tokens, stop_words)
    print(cleaned_tokens)
    if cleaned_tokens is None:
        return
    frequencies = calculate_frequencies(cleaned_tokens)
    print(frequencies)
    if frequencies is None:
        return
    top_n = get_top_n(frequencies, 5)
    print(top_n)
    if top_n is None:
        return
    tf_d = calculate_tf(frequencies)
    print(tf_d)
    if tf_d is None:
        return
    with open("assets/IDF.json", "r", encoding="utf-8") as file:
        idf = load(file)
    idf_d = calculate_tfidf(tf_d, idf)
    print(idf_d)
    if idf_d is None:
        return
    top_idf = get_top_n(idf_d, 10)
    print(top_idf)
    if top_idf is None:
        return
    with open("assets/corpus_frequencies.json", "r", encoding="utf-8") as file:
        corpus_freqs = load(file)
    result = top_idf
    assert result, "Keywords are not extracted"


if __name__ == "__main__":
    main()
