"""
Spellcheck starter
"""

# pylint:disable=unused-variable, duplicate-code, too-many-locals
from lab_1_keywords_tfidf.main import clean_and_tokenize, remove_stop_words
from lab_2_spellcheck.main import build_vocabulary, find_correct_word, find_out_of_vocab_words


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
    alphabet = list("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")
    all_words = clean_and_tokenize(text) or []
    words_no_stop = remove_stop_words(all_words, stop_words) or []
    vocabulary = build_vocabulary(words_no_stop) or {}
    all_text = []
    for sentence in sentences:

        tokens = clean_and_tokenize(sentence) or []
        tokens_no_stop = remove_stop_words(tokens, stop_words) or []
        all_text.extend(tokens_no_stop)
    wrong_words = find_out_of_vocab_words(all_text, vocabulary) or []
    fixed_words = {}
    for token in wrong_words:
        correct_jaccard = find_correct_word(token, vocabulary, "jaccard", alphabet)
        correct_levenshtein = find_correct_word(token, vocabulary, "levenshtein", alphabet)
        correct_frequency = find_correct_word(token, vocabulary, "frequency-based", alphabet)
        correct_jaro_winkler = find_correct_word(token, vocabulary, "jaro-winkler", alphabet)
        fixed_words[token] = {
            "jaccard", correct_jaccard,
            "levenshtein", correct_levenshtein,
            "frequency-based", correct_frequency,
            "jaro-winkler", correct_jaro_winkler
        }
    result = fixed_words
    assert result, "Result is None"


if __name__ == "__main__":
    main()
