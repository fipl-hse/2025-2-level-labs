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
    calculate_tfidf)

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
    result = None
    tokens = clean_and_tokenize(target_text)
    clean_tokens = remove_stop_words(tokens, stop_words)
    frequency_stats = calculate_frequencies(clean_tokens)
    top_freq_tokens = get_top_n(frequency_stats, 10)
    tf_dict = calculate_tf(frequency_stats)
    tfidf_calculated = calculate_tfidf(tf_dict, idf)
    top_tfidf_tokens = get_top_n(tfidf_calculated, 10)
    print(top_tfidf_tokens)
    result = top_tfidf_tokens
    assert result, "Keywords are not extracted"


if __name__ == "__main__":
    main()
