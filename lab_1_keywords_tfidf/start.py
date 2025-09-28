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
    tokens = clean_and_tokenize(target_text) or []
    tokens_without_stopwords = remove_stop_words(tokens, stop_words) or []
    print(tokens_without_stopwords)
    frequencies = calculate_frequencies(tokens_without_stopwords) or {}
    term_freq_tf = calculate_tf(frequencies) or {}
    print(term_freq_tf)
    term_freq_tfidf = calculate_tfidf(term_freq_tf, idf) or {}
    print(term_freq_tfidf)
    top_key_words = get_top_n(term_freq_tfidf, 10) or []
    print(', '.join(top_key_words))
    

if __name__ == "__main__":
    main()
