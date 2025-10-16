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

    cleaned_tokens = clean_and_tokenize(text) or []

    removed_stop_words = remove_stop_words(cleaned_tokens, stop_words) or []

    vocabulary = build_vocabulary(removed_stop_words) or {}

    sentences_cleaned_tokens = list(set(
        token
        for sentence in sentences
        for token in remove_stop_words(clean_and_tokenize(sentence) or [], stop_words)
    ))
    out_of_vocab_words = find_out_of_vocab_words(sentences_cleaned_tokens, vocabulary)
    print("Words out of vocabulary:", out_of_vocab_words, sep="\n")

    alphabet = list("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")
    jaccard_corrections = {}
    frequency_based_corrections = {}
    levenshtein_corrections = {}
    jaro_winkler_corrections= {}
    for word in out_of_vocab_words:
        jaccard_corrections[word] = find_correct_word(word, vocabulary, "jaccard", alphabet)

        frequency_based_corrections[word] = find_correct_word(word, vocabulary, "frequency-based", alphabet)

        levenshtein_corrections[word] = find_correct_word(word, vocabulary, "levenshtein", alphabet)

        jaro_winkler_corrections[word] = find_correct_word(word, vocabulary, "jaro-winkler", alphabet)


    result = [jaccard_corrections,
            frequency_based_corrections,
            levenshtein_corrections,
            jaro_winkler_corrections
    ]
    print(result)
    assert result, "Result is None"

if __name__ == "__main__":
    main()
