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
    tokens = clean_and_tokenize(text)
    tokens_without_stopwords = remove_stop_words(tokens, stop_words)
    print("Tokens without stopwords: ", tokens_without_stopwords)
    if tokens is not None:
         vocabulary = build_vocabulary(tokens_without_stopwords)
    print("Vocabulary with relative frequencies:")
    sorted_vocab = sorted(vocabulary.items(), key=lambda x: x[1], reverse=True)[:10]
    for word, freq in sorted_vocab:
            print(f"{word}: {freq}")
    for i, sentence in enumerate(sentences, 1):
            sentence_tokens = clean_and_tokenize(sentence)
            if sentence_tokens is not None:
                sentence_tokens = remove_stop_words(sentence_tokens, stop_words)
            if sentence_tokens is not None and vocabulary is not None:
                 out_of_vocabulary_words = find_out_of_vocab_words(sentence_tokens, vocabulary)     
    result = out_of_vocabulary_words
    assert result, "Result is None"


if __name__ == "__main__":
    main()
