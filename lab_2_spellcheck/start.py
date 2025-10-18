"""
Spellcheck starter
"""
from lab_1_keywords_tfidf.main import (
    clean_and_tokenize,
    remove_stop_words,
)
from lab_2_spellcheck.main import (
    build_vocabulary,
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
    tokens_text = clean_and_tokenize(text) or []
    tokens_text_without_stop_words = remove_stop_words(tokens_text, stop_words) or []
    vocabulary = build_vocabulary(tokens_text_without_stop_words) or {}

    all_sentence_tokens = []
    for sentence in sentences:
        sentence_tokens = clean_and_tokenize(sentence) or []
        sentence_tokens_without_stop_words = remove_stop_words(sentence_tokens, stop_words) or []
        all_sentence_tokens.extend(sentence_tokens_without_stop_words)
    error_words = find_out_of_vocab_words(all_sentence_tokens, vocabulary) or []
    print(error_words)
    alphabet = [chr(i) for i in range(1072, 1104)]
    all_results = {}
    for error_word in error_words:
        print(f"\nИсправление для '{error_word}':")
        jaccard_correction = find_correct_word(error_word, vocabulary,
                                               'jaccard', alphabet) or {}
        frequency_correction = find_correct_word(error_word, vocabulary,
                                                 'frequency-based', alphabet) or {}
        levenshtein_correction = find_correct_word(error_word, vocabulary,
                                                   'levenshtein', alphabet) or {}
        print(f"  Jaccard: {jaccard_correction}")
        print(f"  Frequency-based: {frequency_correction}")
        print(f"  Levenshtein: {levenshtein_correction}")
        all_results[error_word] = {
            'jaccard': jaccard_correction,
            'frequency-based': frequency_correction,
            'levenshtein': levenshtein_correction,
        }
    result = all_results
    assert result, "Result is None"

if __name__ == "__main__":
    main()
