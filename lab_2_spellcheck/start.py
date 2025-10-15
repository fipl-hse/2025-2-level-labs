"""
Spellcheck starter
"""

# pylint:disable=unused-variable, duplicate-code, too-many-locals
from lab_1_keywords_tfidf.main import (
    clean_and_tokenize,
    remove_stop_words
    )

from lab_2_spellcheck.main import (
    build_vocabulary,
    find_out_of_vocab_words,
    find_correct_word
)


Alphabet = list("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")

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
    
    tokenized_corpus = clean_and_tokenize(text)
    corpus_without_stopwords = remove_stop_words(tokenized_corpus, stop_words)
    vocabulary = build_vocabulary(corpus_without_stopwords)
    tokenized_sentences = clean_and_tokenize(''.join(sentences))
    sentences_without_stopwords = remove_stop_words(tokenized_sentences, stop_words)
    tokens_out_of_voc = find_out_of_vocab_words(sentences_without_stopwords, vocabulary)
    
    
    result = {find_correct_word(token, vocabulary, "frequency-based", Alphabet): token for token in tokens_out_of_voc}
    print(result)
    assert result, "Result is None"


if __name__ == "__main__":
    main()
