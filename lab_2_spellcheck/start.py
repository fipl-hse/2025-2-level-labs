"""
Spellcheck starter
"""

# pylint:disable=unused-variable, duplicate-code, too-many-locals

from lab_1_keywords_tfidf.main import clean_and_tokenize, remove_stop_words
from lab_2_spellcheck.main import (
    build_vocabulary,
    find_correct_word,
    find_out_of_vocab_words,
)


def main() -> None:
    """
    Launches an implementation.
    """
    text_without_stop_words = None
    vocabulary = None
    absent_words = None
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
    alphabet = list("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")
    if cleaned_text is not None:
        text_without_stop_words = remove_stop_words(cleaned_text, stop_words)
    cleaned_sentences = []
    for sentence in sentences:
        tokenized_sentence = clean_and_tokenize(sentence)
        if tokenized_sentence is not None:
            cleaned_sentences.extend(tokenized_sentence)
    sentences_without_stop_words = remove_stop_words(cleaned_sentences, stop_words)
    if text_without_stop_words is not None:
        vocabulary = build_vocabulary(text_without_stop_words)
        if vocabulary is not None:
            absent_words = find_out_of_vocab_words(sentences_without_stop_words, vocabulary)
    final_result = {}
    if (absent_words is not None
    and vocabulary is not None):
        for word in absent_words:
            jaccard_result = find_correct_word(word, vocabulary, "jaccard", alphabet)
            frequency_result = find_correct_word(word, vocabulary, "frequency-based", alphabet)
            levenshtein_result = find_correct_word(word, vocabulary, "levenshtein", alphabet)
            jaro_winkler_result = find_correct_word(word, vocabulary, "jaro-winkler", alphabet)
            print(f"jaccard: {jaccard_result}")
            print(f"frequency_based: {frequency_result}")
            print(f"levenshtein: {levenshtein_result}")
            print(f"jaro_winkler: {jaro_winkler_result}")
            final_result[word] = {
                "jaccard": jaccard_result,
                "frequency_based": frequency_result,
                "levenshtein": levenshtein_result,
                "jaro_winkler": jaro_winkler_result
            }
    result = final_result
    assert result, "Result is None"


if __name__ == "__main__":
    main()
