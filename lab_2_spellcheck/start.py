"""
Spellcheck starter
"""

# pylint:disable=unused-variable, duplicate-code, too-many-locals

from lab_1_keywords_tfidf.main import clean_and_tokenize, remove_stop_words
from lab_2_spellcheck.main import (
    build_vocabulary,
    calculate_distance,
    find_out_of_vocab_words,
)


def main() -> None:
    """
    Launches an implementation.
    """
    text_without_stop_words = None
    vocabulary = None
    absent_words = None
    first_result = None
    with open("assets/Master_and_Margarita_chapter1.txt", "r", encoding="utf-8") as file:
        text = file.read()
    with open("assets/stop_words.txt", "r", encoding="utf-8") as file:
        stop_words = file.read().split("\n")
    with (
        open("assets/incorrect_sentence_1.txt", "r", encoding="utf-8") as f1,
        open("assets/incorrect_sentence_2.txt", "r", encoding="utf-8") as f2,
        open("assets/incorrect_sentence_3.txt", "r", encoding="utf-8") as f3,
        open("assets/incorrect_sentence_4.txt", "r", encoding="utf-8") as f4,
        open("assets/incorrect_sentence_5.txt", "r", encoding="utf-8") as f5,
    ):
        sentences = [f.read() for f in (f1, f2, f3, f4, f5)]
    cleaned_text = clean_and_tokenize(text)
    if cleaned_text is not None:
        text_without_stop_words = remove_stop_words(cleaned_text, stop_words)
    if text_without_stop_words is not None:
        vocabulary = build_vocabulary(sentences)
        if vocabulary is not None:
            absent_words = find_out_of_vocab_words(text_without_stop_words, vocabulary)
    if (absent_words is not None
    and vocabulary is not None):
        for word in absent_words:
            first_result = calculate_distance(word, vocabulary, "jaro-winkler")
    if first_result is not None:
        print(first_result)
    result = first_result
    assert result, "Result is None"


if __name__ == "__main__":
    main()
