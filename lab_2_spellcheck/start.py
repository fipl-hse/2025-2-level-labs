"""
Spellcheck starter
"""

# pylint:disable=unused-variable, duplicate-code, too-many-locals
from lab_2_spellcheck.main import(
    build_vocabulary,
    calculate_distance,
    clean_and_tokenize,
    find_out_of_vocab_words,
    remove_stop_words
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

    tokens = clean_and_tokenize(text) or []
    tokens_without_stopwords = remove_stop_words(tokens, stop_words) or []
    tokens_dict = build_vocabulary(tokens_without_stopwords) or {}
    print(tokens_dict)
    out_of_vocab_words = find_out_of_vocab_words(tokens, tokens_dict) or []
    print(out_of_vocab_words)
    jaccard_distance = []
    for word in out_of_vocab_words:
        jaccard_distance.append(calculate_distance(word, tokens_dict, "jaccard"))
    print(jaccard_distance)
    result = jaccard_distance
    assert result, "Result is None"


if __name__ == "__main__":
    main()
