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
    propose_candidates,
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
    tokens_without_stop_words = remove_stop_words(tokens, stop_words) or []
    vocabulary = build_vocabulary(tokens_without_stop_words) or {}
    out_of_vocab_words = find_out_of_vocab_words(tokens_without_stop_words, vocabulary) or []
    alphabet = [chr(i) for i in range(1072, 1104)]
    methods = ("jaccard", "frequency-based", "levenshtein", "jaro-winkler")
    result_ = {}
    for word in out_of_vocab_words:
        word_result = {}
        for method in ("jaccard", "frequency-based", "levenshtein", "jaro-winkler"):
            correction = find_correct_word(word, vocabulary, method, alphabet)
            word_result[method] = correction
        result_[word] = word_result
    print(result_)
    result = result_
    assert result, "Result is None"


if __name__ == "__main__":
    main()
