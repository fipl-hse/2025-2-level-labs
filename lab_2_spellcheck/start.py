"""
Spellcheck starter
"""

# pylint:disable=unused-variable, duplicate-code, too-many-locals
from lab_1_keywords_tfidf.main import clean_and_tokenize, remove_stop_words
from lab_2_spellcheck.main import build_vocabulary, find_correct_word, find_out_of_vocab_words

russian_alphabet = list("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")

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

    text_tokenized = clean_and_tokenize(text)
    if not text_tokenized:
        return
    stopwords_removed = remove_stop_words(text_tokenized, stop_words)
    if not stopwords_removed:
        return
    sentence_tokenized = clean_and_tokenize(''.join(sentences))
    if not sentence_tokenized:
        return
    sentence_stopwords_removed = remove_stop_words(sentence_tokenized, stop_words)
    if not sentence_stopwords_removed:
        return
    vocabulary = build_vocabulary(stopwords_removed)
    if not vocabulary:
        return
    print("Vocabulary: ", vocabulary, "\n")
    vocabulary_tokens = find_out_of_vocab_words(sentence_stopwords_removed, vocabulary)
    if not vocabulary_tokens:
        return
    print("Tokens out of vocabulary: ", vocabulary_tokens, "\n")
    freq_method = {token: find_correct_word(
        token, vocabulary, "frequency-based", russian_alphabet)
        for token in vocabulary_tokens}
    print("Correct words by frequensy based method: ", freq_method, "\n")
    jaccard_method = {token: find_correct_word(
        token, vocabulary, "jaccard", russian_alphabet)
        for token in vocabulary_tokens}
    print("Correct words by jaccard method: ", jaccard_method, "\n")
    levenshtein_method = {token: find_correct_word(
        token, vocabulary, "levenshtein", russian_alphabet)
        for token in vocabulary_tokens}
    print("Correct words by levenshtein method: ", levenshtein_method, "\n")
    jaro_winkler_method = {token: find_correct_word(
        token, vocabulary, "jaro-winkler", russian_alphabet)
        for token in vocabulary_tokens}
    print("Correct words by jaro-winkler method: ", jaro_winkler_method, "\n")
    result = jaro_winkler_method
    assert result, "Result is None"


if __name__ == "__main__":
    main()
