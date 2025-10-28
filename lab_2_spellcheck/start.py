"""
Spellcheck starter
"""

# pylint:disable=unused-variable, duplicate-code, too-many-locals
from lab_1_keywords_tfidf.main import clean_and_tokenize, remove_stop_words
from lab_2_spellcheck.main import (
    build_vocabulary,
    calculate_distance,
    calculate_frequency_distance,
    calculate_jaro_winkler_distance,
    calculate_levenshtein_distance,
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
    tokens_vocab = build_vocabulary(tokens_without_stopwords) or {}
    print(tokens_vocab)

    tokens_not_in_vocab = find_out_of_vocab_words(tokens_without_stopwords, tokens_vocab) or []
    print(tokens_not_in_vocab)

    jaccard_distance = calculate_distance("кот", {"кот": 0.5, "пёс": 0.5},
                                                 method = "jaccard") or {}
    print(jaccard_distance)

    alphabet = list("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")
    freq_distances = calculate_frequency_distance("маладой", tokens_vocab, alphabet) or {}
    print(freq_distances)

    levenshtein_distance = calculate_levenshtein_distance("кот", "кто")
    print(levenshtein_distance)

    jaro_winkler_distance = calculate_jaro_winkler_distance("кот", "кто")
    print(jaro_winkler_distance)
    result = jaro_winkler_distance

    all_wrong_words = []
    for sentence in sentences:
        sentence_tokens = clean_and_tokenize(sentence) or []
        out_of_vocab = find_out_of_vocab_words(sentence_tokens, tokens_vocab) or []
        all_wrong_words.extend(out_of_vocab)
    unique_wrong_words = sorted(set(all_wrong_words))

    for wrong_word in unique_wrong_words:
        print(f"Исправления для слова '{wrong_word}':")
        correct_word = find_correct_word(wrong_word, tokens_vocab, "jaccard", alphabet)
        if correct_word and correct_word != wrong_word:
            print(f"jaccard: {correct_word}")
        correct_word = find_correct_word(wrong_word, tokens_vocab, "frequency-based", alphabet)
        if correct_word and correct_word != wrong_word:
            print(f"frequency-based: {correct_word}")
        correct_word = find_correct_word(wrong_word, tokens_vocab, "levenshtein", alphabet)
        if correct_word and correct_word != wrong_word:
            print(f"levenshtein: {correct_word}")
        correct_word = find_correct_word(wrong_word, tokens_vocab, "jaro-winkler", alphabet)
        if correct_word and correct_word != wrong_word:
            print(f"jaro-winkler: {correct_word}")
    assert result, "Result is None"


if __name__ == "__main__":
    main()
