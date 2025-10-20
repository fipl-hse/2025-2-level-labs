"""
Spellcheck starter
"""
from lab_1_keywords_tfidf.main import clean_and_tokenize, remove_stop_words
from lab_2_spellcheck.main import (
    build_vocabulary,
    find_correct_word,
    find_out_of_vocab_words,
)

# pylint:disable=unused-variable, duplicate-code, too-many-locals


def main() -> None:
    """
    Launches an implementation.
    """
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
    tokens = clean_and_tokenize(text) or []
    tokens_without_stopwords = remove_stop_words(tokens, stop_words) or []
    vocabulary = build_vocabulary(tokens_without_stopwords) or {}
    all_sentences_wrong_words = []
    for sentence in sentences:
        sentence_tokens = clean_and_tokenize(sentence) or []
        sentence_tokens_without_stopwords = remove_stop_words(sentence_tokens, stop_words) or []
        out_of_vocab_words = find_out_of_vocab_words(sentence_tokens_without_stopwords, vocabulary)
        all_sentences_wrong_words.extend(out_of_vocab_words)
    possible_correct_words = {}
    for wrong_word in all_sentences_wrong_words:
        jaccard_correct_word = find_correct_word(wrong_word, vocabulary, "jaccard") or {}
        frequency_based_correct_word = find_correct_word(
            wrong_word, vocabulary, "frequency-based", list("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")
            ) or {}
        levenshtein_correct_word = find_correct_word(
            wrong_word, vocabulary, "levenshtein", list("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")
            ) or {}
        possible_correct_words[wrong_word] = [jaccard_correct_word, frequency_based_correct_word, levenshtein_correct_word]
    result = possible_correct_words
    print(result)
    assert result, "Result is None"


if __name__ == "__main__":
    main()
