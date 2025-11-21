"""
Spellcheck starter
"""

import os

# pylint:disable=unused-variable, duplicate-code, too-many-locals
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)
from lab_1_keywords_tfidf.main import (
    clean_and_tokenize,
    remove_stop_words,
)
from main import (
    build_vocabulary,
    find_correct_word,
    find_out_of_vocab_words,
)


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
    alphabet = list("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")
    for i, sentence in enumerate(sentences, 1):
        print(f"\n--- Предложение {i}: {sentence.strip()} ---")
        sentence_tokens = clean_and_tokenize(sentence) or []
        out_of_vocab = find_out_of_vocab_words(sentence_tokens, vocabulary) or []
        print(f"Слова вне словаря: {out_of_vocab}")
        for token in out_of_vocab:
            correct_word_jaccard = find_correct_word(token, vocabulary, "jaccard") or ""
            correct_word_freq = find_correct_word(token, vocabulary, "frequency-based", alphabet) or ""
            print(f"  Слово '{token}':")
            print(f"    Jaccard исправление: {correct_word_jaccard}")
            print(f"    Frequency-based исправление: {correct_word_freq}")
        result = out_of_vocab
        assert result, "Keywords are not extracted"

if __name__ == "__main__":
    main()