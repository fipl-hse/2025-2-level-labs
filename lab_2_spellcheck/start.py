"""
Spellcheck starter
"""
from lab_1_keywords_tfidf.main import clean_and_tokenize, remove_stop_words
from lab_2_spellcheck.main import (
    build_vocabulary,
    calculate_distance,
    calculate_jaccard_distance,
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
    relative_frequencies = build_vocabulary(tokens_without_stopwords) or {}
    out_of_vocab_words = find_out_of_vocab_words(tokens_without_stopwords, relative_frequencies)
    result = find_correct_word(
         "висною", relative_frequencies, 'levenshtein', list("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")
    )
    assert result, "Result is None"


if __name__ == "__main__":
    main()
