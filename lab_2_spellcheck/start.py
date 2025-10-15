"""
Spellcheck starter
"""

# pylint:disable=unused-variable, duplicate-code, too-many-locals
from lab_1_keywords_tfidf.main import (
    clean_and_tokenize,
    remove_stop_words,
)
from lab_2_spellcheck.main import (
    build_vocabulary,
    calculate_distance,
    calculate_frequency_distance,
    calculate_levenshtein_distance,
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
    result = None
    tokens = clean_and_tokenize(text) or []
    without_stop_words = remove_stop_words(tokens, stop_words) or []
    tokens_vocabulary = build_vocabulary(without_stop_words) or {}
    print(tokens_vocabulary)
    tokens_not_in_vocab = find_out_of_vocab_words(without_stop_words, tokens_vocabulary) or []
    print(tokens_not_in_vocab)
    jaccard_distance = calculate_distance("кот", {"кот": 0.5, "пёс": 0.5},
                                                 method = "jaccard") or {}
    print(jaccard_distance)
    alphabet = list("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")
    freq_distances = calculate_frequency_distance("маладой", tokens_vocabulary, alphabet) or {}
    print(freq_distances)
    levenshtein_distance = calculate_levenshtein_distance("кот", "кто")
    print(levenshtein_distance)
    result = levenshtein_distance
    assert result, "Result is None"


if __name__ == "__main__":
    main()
