"""
Frequency-driven keyword extraction starter
"""

# pylint:disable=too-many-locals, unused-argument, unused-variable, invalid-name, duplicate-code
from json import load

from lab_1_keywords_tfidf.main import (
    clean_and_tokenize,
    remove_stop_words,
    calculate_frequencies,
    get_top_n,
    calculate_tf,
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
    text_tokenized = clean_and_tokenize(target_text) or []
    stopwords_removed = remove_stop_words(text_tokenized, stop_words) or []
    freq_dict = calculate_frequencies(stopwords_removed) or {}
    tf_dict = calculate_tf(freq_dict) or {}
    tfidf_dict = calculate_tfidf(tf_dict, idf) or {}
    significant_list = get_top_n(tfidf_dict, 10) or []
    print(', '.join(significant_list))
    result = None
    assert result, "Keywords are not extracted"


if __name__ == "__main__":
    main()
