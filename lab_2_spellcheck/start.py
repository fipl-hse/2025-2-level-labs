"""
Spellcheck starter
"""

# pylint:disable=unused-variable, duplicate-code, too-many-locals
from lab_1_keywords_tfidf.main import(
    calculate_frequencies,
    remove_stop_words,
    clean_and_tokenize,
)

from lab_2_spellcheck.main import(
    build_vocabulary,
    find_out_of_vocab_words,
    calculate_jaccard_distance,
    calculate_distance
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

    # tokenized_text = clean_and_tokenize(text)

    # removed_stop_words = remove_stop_words(tokenized_text, stop_words)

    # built_vocabulary = build_vocabulary(removed_stop_words)

    # out_of_vocab_words = find_out_of_vocab_words(removed_stop_words, built_vocabulary)

    # result = calculate_distance("кот", {"кот": 0.5, "пёс": 0.5}, method="jaccard")
    # print(result)
    # print(built_vocabulary)


    print(min(["abs", "ff", "gg", "pmg"]))


    result = None
    # assert result, "Result is None"


if __name__ == "__main__":
    main()
