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
    tokenized_text = clean_and_tokenize(target_text) or []
    text_without_stopwords = remove_stop_words(tokenized_text, stop_words) or []
    frequ_dict = calculate_frequencies(text_without_stopwords) or {}
    tf_dict = calculate_tf(frequ_dict) or {}
    tfidf_dict = calculate_tfidf(tf_dict, idf) or {}
    top_n_tfidf = get_top_n(tfidf_dict, 10)
    result = top_n_tfidf

    print(result)
    assert result, "Keywords are not extracted"


if __name__ == "__main__":
    main()
