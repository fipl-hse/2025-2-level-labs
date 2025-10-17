"""
Spellcheck starter
"""

# pylint:disable=unused-variable, duplicate-code, too-many-locals
from lab_1_keywords_tfidf.main import clean_and_tokenize, remove_stop_words
from lab_2_spellcheck.main import (
    build_vocabulary, calculate_distance, find_out_of_vocab_words, find_correct_word 
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
        tokens_no_stop = remove_stop_words(tokens, stop_words) or []
        frequencies = build_vocabulary(tokens_no_stop) or {}
        found_by_jaccard_word = find_correct_word("кит", {"кот": 0.5, "пёс": 0.5}, method = "jaccard")
        found_by_frequency_word = find_correct_word("пиро", frequencies, 
                                                    "frequency-based", list("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")) #пиво
        found_by_levenshtein_word = find_correct_word("нага", frequencies, 
                                                      "levenshtein") #наша
        found_by_jaro_winkler_word = find_correct_word("вторый", frequencies, 
                                                       "jaro-winkler") #второй
    result = found_by_jaro_winkler_word
    assert result, "Result is None"


if __name__ == "__main__":
    main()
