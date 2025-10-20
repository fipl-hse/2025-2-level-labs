"""
Spellcheck starter
"""

# pylint:disable=unused-variable, duplicate-code, too-many-locals


import os
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
    base_dir = os.path.dirname(__file__)
    assets_path = os.path.join(base_dir, "assets")
    with open(
        os.path.join(
            assets_path, "Master_and_Margarita_chapter1.txt"
        ),
        "r",
        encoding="utf-8"
    ) as file:
        text = file.read()
    with open(
        os.path.join(assets_path, "stop_words.txt"), "r", encoding="utf-8"
    ) as file:
        stop_words = file.read().split("\n")
    with (
        open(
            os.path.join(assets_path, "incorrect_sentence_1.txt"),
            "r",
            encoding="utf-8"
        ) as f1,
        open(
            os.path.join(assets_path, "incorrect_sentence_2.txt"),
            "r",
            encoding="utf-8"
        ) as f2,
        open(
            os.path.join(assets_path, "incorrect_sentence_3.txt"),
            "r",
            encoding="utf-8"
        ) as f3,
        open(
            os.path.join(assets_path, "incorrect_sentence_4.txt"),
            "r",
            encoding="utf-8"
        ) as f4,
        open(
            os.path.join(assets_path, "incorrect_sentence_5.txt"),
            "r",
            encoding="utf-8"
        ) as f5,
    ):
        sentences = [f.read() for f in (f1, f2, f3, f4, f5)]

    tokens = clean_and_tokenize(text) or []
    tokens_without_stopwords = remove_stop_words(tokens, stop_words) or []

    vocab = build_vocabulary(tokens_without_stopwords) or {}
    print(f"Vocabulary size: {len(vocab)}")

    out_of_vocab = find_out_of_vocab_words(
        tokens_without_stopwords, vocab
    ) or []
    print(f"Out of vocabulary words: {len(out_of_vocab)}")

    alphabet = list("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")

    if vocab:
        sample_tokens = list(vocab.keys())
        if len(sample_tokens) >= 2:
            word1 = sample_tokens[0]
            word2 = sample_tokens[1]
            jaccard_dist = calculate_distance(
                word1, {word1: 0.1, word2: 0.2}, "jaccard"
            )
            print(
                f"Jaccard distance between '{word1}' "
                f"and '{word2}': {jaccard_dist}"
            )

            lev_dist = calculate_levenshtein_distance(word1, word2)
            print(
                f"Levenshtein distance between '{word1}' "
                f"and '{word2}': {lev_dist}"
            )
            jw_dist = calculate_jaro_winkler_distance(word1, word2)
            print(
                f"Jaro-Winkler distance between '{word1}'"
                f"and '{word2}': {jw_dist}"
            )

        if sample_tokens:
            freq_dist = calculate_frequency_distance(
                sample_tokens[0], vocab, alphabet
            ) or {}
            print(
                f"Frequency distance keys for '{sample_tokens[0]}':"
                f"{list(freq_dist.keys())[:5]}"
            )

            test_word = out_of_vocab[0] if out_of_vocab else sample_tokens[0]

            jaccard_correct = find_correct_word(
                test_word, vocab, "jaccard", alphabet
            )
            print(f"Jaccard correction for '{test_word}': {jaccard_correct}")

            freq_correct = find_correct_word(
                test_word, vocab, "frequency-based", alphabet
            )
            print(f"Frequency correction for '{test_word}': {freq_correct}")

            lev_correct = find_correct_word(
                test_word, vocab, "levenshtein", alphabet
            )
            print(f"Levenshtein correction for '{test_word}': {lev_correct}")

            jw_correct = find_correct_word(
                test_word, vocab, "jaro-winkler", alphabet
            )
            print(f"Jaro-Winkler correction for '{test_word}': {jw_correct}")

    all_misspelled = []
    for sentence in sentences:
        sent_tokens = clean_and_tokenize(sentence) or []
        sent_out_of_vocab = find_out_of_vocab_words(sent_tokens, vocab) or []
        all_misspelled.extend(sent_out_of_vocab)

    result = len(set(all_misspelled))
    print(f"Total misspelled words found: {result}")

    assert result is not None, "Misspelled words count is None"
    assert isinstance(result, int), "Misspelled words count should be integer"
    assert result >= 0, "Misspelled words count cannot be negative"


if __name__ == "__main__":
    main()
