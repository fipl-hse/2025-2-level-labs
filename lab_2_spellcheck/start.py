"""
Spellcheck starter
"""

# pylint:disable=unused-variable, duplicate-code, too-many-locals
from lab_1_keywords_tfidf.main import clean_and_tokenize, remove_stop_words
from lab_2_spellcheck.main import (
    build_vocabulary,
    calculate_distance,
    find_correct_word,
    find_out_of_vocab_words
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

    alphabet = list(map(chr, range(97, 123)))

    tokens = clean_and_tokenize(text) or []
    tokens_without_stopwords = remove_stop_words(tokens, stop_words) or []
    tokens_dict = build_vocabulary(tokens_without_stopwords) or {}
    out_of_vocab_words = find_out_of_vocab_words(tokens, tokens_dict) or []

    jaccard_distance = []
    for word in out_of_vocab_words:
        jaccard_distance.append(calculate_distance(word, tokens_dict, "jaccard"))
    print(jaccard_distance)

    frequency_based_correct_word = []
    for word in out_of_vocab_words:
        frequency_based_correct_word.append(find_correct_word(word, tokens_dict,
                                                            "frequency-based", alphabet))
    print(frequency_based_correct_word)

    levenshtein_correct_word = []
    for word in out_of_vocab_words:
        levenshtein_correct_word.append(find_correct_word(word, tokens_dict, "levenshtein"))
        
    result = jaccard_distance
    assert result, "Result is None"


if __name__ == "__main__":
    main()
