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
    with open(
        "assets/Master_and_Margarita_chapter1.txt", "r", encoding="utf-8"
    ) as file:
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
    tokens_without_stopwords = remove_stop_words(tokens, stop_words) or []

    vocab = build_vocabulary(tokens_without_stopwords) or {}
    print(f"Vocabulary size: {len(vocab)}")

    out_of_vocab = find_out_of_vocab_words(
        tokens_without_stopwords, vocab
    ) or []
    print(f"Out of vocabulary words: {len(out_of_vocab)}")

    alphabet = list("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")

    jaccard_dist = calculate_distance(
        "кот", {"кот": 0.1, "кто": 0.2}, "jaccard"
    )
    print(f"Jaccard distance: {jaccard_dist}")

    freq_dist = calculate_frequency_distance("маладой", vocab, alphabet) or {}
    print(f"Frequency distance keys: {list(freq_dist.keys())[:5]}")

    lev_dist = calculate_levenshtein_distance("кот", "кто")
    print(f"Levenshtein distance: {lev_dist}")

    jw_dist = calculate_jaro_winkler_distance("кот", "кто")
    print(f"Jaro-Winkler distance: {jw_dist}")

    test_word = "маладой"

    jaccard_correct = find_correct_word(
        test_word, vocab, "jaccard", alphabet
    )
    print(f"Jaccard correction for '{test_word}': {jaccard_correct}")

    freq_correct = find_correct_word(
        test_word, vocab, "frequency-based", alphabet
    )
    print(f"Frequency correction for '{test_word}': {freq_correct}")

    lev_correct = find_correct_word(test_word, vocab, "levenshtein", alphabet)
    print(f"Levenshtein correction for '{test_word}': {lev_correct}")

    jw_correct = find_correct_word(test_word, vocab, "jaro-winkler", alphabet)
    print(f"Jaro-Winkler correction for '{test_word}': {jw_correct}")

    all_misspelled = []
    for sentence in sentences:
        sent_tokens = clean_and_tokenize(sentence) or []
        sent_out_of_vocab = find_out_of_vocab_words(sent_tokens, vocab) or []
        all_misspelled.extend(sent_out_of_vocab)

    print(f"Total misspelled words found: {len(set(all_misspelled))}")

    result = jw_dist

    assert result is not None, "Result is None"


if __name__ == "__main__":
    main()
