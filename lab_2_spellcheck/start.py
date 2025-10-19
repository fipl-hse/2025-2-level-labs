"""
Spellcheck starter
"""

# pylint:disable=unused-variable, duplicate-code, too-many-locals
from typing import Literal

from lab_1_keywords_tfidf.main import (
    clean_and_tokenize,
    remove_stop_words,
)
from lab_2_spellcheck.main import (
    build_vocabulary,
    calculate_distance,
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
    russian_alphabet = list('абвгдеёжзийклмнопрстуфхцчшщъыьэюя')
    tokens = clean_and_tokenize(text)
    if not tokens:
        return
    tokens_without_stopwords = remove_stop_words(tokens, stop_words)
    if not tokens_without_stopwords:
        return
    vocabulary = build_vocabulary(tokens_without_stopwords)
    if not vocabulary:
        return
    print("Top-5 words with relative frequency:")
    top_words = sorted(vocabulary.items(), key=lambda x: x[1], reverse=True)[:5]
    for word, freq in top_words:
        print(f"{word}: {freq:.4f}")
    test_token_jaccard = "кат"
    test_vocab_jaccard = {"кот": 0.5, "котик": 0.3, "пес": 0.2}
    jaccard_distances = calculate_distance(test_token_jaccard, test_vocab_jaccard, method="jaccard")
    print(f"Jaccard distance for '{test_token_jaccard}'")
    if jaccard_distances:
        print(f"Dictionary of distances: {jaccard_distances}")
    sentence = sentences[0]
    print("\nSentence 1")
    print(f"Original: {sentence}")
    sentence_tokens = clean_and_tokenize(sentence)
    if not sentence_tokens:
        return
    sentence_tokens_without_stopwords = remove_stop_words(sentence_tokens, stop_words)
    if not sentence_tokens_without_stopwords:
        return
    oov_words = find_out_of_vocab_words(sentence_tokens_without_stopwords, vocabulary) or []
    if oov_words:
        print(f"Out-of-vocabulary words: {oov_words}")
    for wrong_word in oov_words:
        print(f"\nProcessing word: '{wrong_word}'")
        methods_tuple: tuple[
        Literal["jaccard", "frequency-based", "levenshtein", "jaro-winkler"], ...] = (
        "jaccard", "frequency-based", "levenshtein", "jaro-winkler"
        )
        for method in methods_tuple:
            distances = calculate_distance(wrong_word, vocabulary, method, russian_alphabet)
            correction = find_correct_word(wrong_word, vocabulary, method, russian_alphabet)
            if not distances:
                print(f"{method}: Failed to calculate distances")
                continue
            top_candidates = sorted(distances.items(), key=lambda x: x[1])[:3]
            print(f"{method}: '{wrong_word}' -> '{correction}'")
            print(f"Top 3 candidates: {top_candidates}")
            if method == "levenshtein" and correction:
                distance = calculate_levenshtein_distance(wrong_word, correction)
                print(f"Final Levenshtein distance: {distance}")
    result = distance
    assert result, "Result is None"


if __name__ == "__main__":
    main()
