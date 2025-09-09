"""
Frequency-driven keyword extraction starter
"""

# pylint:disable=too-many-locals, unused-argument, unused-variable, invalid-name, duplicate-code
from json import load
from main import clean_and_tokenize, remove_stop_words, calculate_frequencies, get_top_n, calculate_tf, calculate_tfidf

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
    cleaned_tokens = clean_and_tokenize(target_text)
    # print(cleaned_tokens)
    removed_stop_words = remove_stop_words(cleaned_tokens, stop_words)
    # print(removed_stop_words)
    calculated_frequencies = calculate_frequencies(removed_stop_words)
    # print(calculated_frequencies)
    top_words = get_top_n(calculated_frequencies, 5)
    # print(top_words)
    term_frequency = calculate_tf(calculated_frequencies)
    # print(term_frequency)
    tf_tdf = calculate_tfidf(term_frequency, idf)
    # print(tf_tdf)
    top_words = get_top_n(tf_tdf, 10)
    # print(top_words)
    # result = None
    # assert result, "Keywords are not extracted"


if __name__ == "__main__":
    main()
