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
    calculate_jaccard_distance,
    calculate_distance,
    find_correct_word,
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
    result = None
    cleaned_and_tokenized_text=clean_and_tokenize(text)
    if cleaned_and_tokenized_text is None:
        return

    text_without_stop_words=remove_stop_words(cleaned_and_tokenized_text, stop_words)
    if text_without_stop_words is None:
        return
    
    vocabulary=build_vocabulary(text_without_stop_words)
    if vocabulary is None:
        return
    print("vocabulary",vocabulary)

    finded_out_of_vocab_words=find_out_of_vocab_words(text_without_stop_words, vocabulary)
    if finded_out_of_vocab_words is None:
        return
    print("yhghyt", finded_out_of_vocab_words)

    '''jaccard_distance=calculate_jaccard_distance(finded_out_of_vocab_words, candidate)
    if jaccard_distance is None:
        return

    calculated_distance=calculate_distance(jaccard_distance)
    if calculated_distance is None:
        return 
    print ("Distance: ", calculated_distance)

    finded_correct_word=find_correct_word(calculate_distance)
    if finded_correct_word is None:
        return'''

    result=finded_out_of_vocab_words
    assert result, "Result is None"


if __name__ == "__main__":
    main()
