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
    calculate_distance,
    calculate_jaccard_distance,
    find_out_of_vocab_words,
    find_correct_word
)

def main() -> None:
    """
    Launches an implementation.
    """
    first_result = None
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
    cleaned_text = clean_and_tokenize(text)
    text_without_stop_words = remove_stop_words(cleaned_text, stop_words)
    vocabulary = build_vocabulary(sentences)
    absent_words = find_out_of_vocab_words(text_without_stop_words, vocabulary)
    for word in absent_words:
        first_result = calculate_distance(word, vocabulary, "jaccard")
    if first_result is not None:
        print(first_result)
    result = first_result
    assert result, "Result is None"


if __name__ == "__main__":
    main()
