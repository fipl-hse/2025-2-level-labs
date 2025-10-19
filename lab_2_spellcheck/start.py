"""
Spellcheck starter
"""

# pylint:disable=unused-variable, duplicate-code, too-many-locals
from lab_1_keywords_tfidf.main import clean_and_tokenize, remove_stop_words
from lab_2_spellcheck.main import build_vocabulary, find_out_of_vocab_words, find_correct_word


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
    tokens = clean_and_tokenize(text)
    if tokens is None:
        tokens = []
    tokens_no_stop = remove_stop_words(tokens, stop_words)
    if tokens_no_stop is None:
        tokens_no_stop = []
    vocabulary = build_vocabulary(tokens_no_stop)
    if vocabulary is None:
        vocabulary = {}
    wrong_words = find_out_of_vocab_words(tokens_no_stop, vocabulary)
    if wrong_words is None:
        wrong_words = []
    alphabet = [chr(i) for i in range(1072, 1104)]
    result = {}
    for word in wrong_words:
        word_results = {}
        jaccard = find_correct_word(word, vocabulary, 'jaccard', alphabet)
        if jaccard is None:
            jaccard = {}
        word_results['jaccard'] = jaccard
        frequency = find_correct_word(word, vocabulary, 'frequency-based', alphabet)
        if frequency is None:
            frequency = {}
        word_results['frequency-based'] = frequency
        levenshtein = find_correct_word(word, vocabulary, 'levenshtein', alphabet)
        if levenshtein is None:
            levenshtein = {}
        word_results['levenshtein'] = levenshtein
        result[word] = word_results
        assert result, "Result is None"
if __name__ == "__main__":
    main()
