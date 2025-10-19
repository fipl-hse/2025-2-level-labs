"""
Spellcheck starter
"""

# pylint:disable=unused-variable, duplicate-code, too-many-locals
from lab_1_keywords_tfidf.main import clean_and_tokenize, remove_stop_words
from lab_2_spellcheck.main import build_vocabulary, find_correct_word, find_out_of_vocab_words

russian = list("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")

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
    corpus_tokens = clean_and_tokenize(text)
    if not corpus_tokens:
        return
    corpus_without_stop_words = remove_stop_words(corpus_tokens, stop_words)
    if not corpus_without_stop_words:
        return
    sentences_tokens = clean_and_tokenize(''.join(sentences))
    if not sentences_tokens:
        return
    sentences_without_stop_words = remove_stop_words(sentences_tokens, stop_words)
    if not sentences_without_stop_words:
        return
    vocabulary = build_vocabulary(corpus_without_stop_words)
    if not vocabulary:
        return
    print("Vocabulary: ", vocabulary, "\n")
    tokens_out_of_vocab = find_out_of_vocab_words(sentences_without_stop_words, vocabulary)
    if not tokens_out_of_vocab:
        return
    print("Tokens out of vocabulary: ", tokens_out_of_vocab, "\n")
    correct_words_jaccard = {token: find_correct_word(
        token, vocabulary, "jaccard", russian)
        for token in tokens_out_of_vocab}
    print("Correct words by jaccard method: ", correct_words_jaccard, "\n")
    correct_words_freq = {token: find_correct_word(
        token, vocabulary, "frequency-based", russian)
        for token in tokens_out_of_vocab}
    print("Correct words by frequensy based method: ", correct_words_freq, "\n")
    correct_words_lev = {token: find_correct_word(
        token, vocabulary, "levenshtein", russian)
        for token in tokens_out_of_vocab}
    print("Correct words by levenshtein method: ", correct_words_lev, "\n")
    correct_words_jaro = {token: find_correct_word(
        token, vocabulary, "jaro-winkler", russian)
        for token in tokens_out_of_vocab}
    print("Correct words by jaro-winkler method: ", correct_words_jaro, "\n")
    result = correct_words_jaro
    result = None
    assert result, "Result is None"


if __name__ == "__main__":
    main()
