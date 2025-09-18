"""
Frequency-driven keyword extraction starter
"""

# pylint:disable=too-many-locals, unused-argument, unused-variable, invalid-name, duplicate-code
from json import load
from main import check_dict, check_float, check_list, check_positive_int, clean_and_tokenize, remove_stop_words, calculate_frequencies, get_top_n


def main() -> None:
    """
    Launches an implementation.
    """
    with open("assets/Дюймовочка.txt", "r", encoding="utf-8") as file:
        target_text = file.read()
    with open("assets/stop_words.txt", "r", encoding="utf-8") as file:
        stop_words = file.read().split("\n")
    with open("assets/IDF.json", "r", encoding="utf-8") as file:
        idf = load(file)
    with open("assets/corpus_frequencies.json", "r", encoding="utf-8") as file:
        corpus_freqs = load(file)
    result = None

    text_tokenized = clean_and_tokenize(target_text)
    tokens_without_stop_words = remove_stop_words(text_tokenized, stop_words)
    frequencies_dictionary = calculate_frequencies(tokens_without_stop_words)
    top_n_tokens = get_top_n(frequencies_dictionary, 10)

    print(top_n_tokens)
    assert result, "Keywords are not extracted"


if __name__ == "__main__":
    main()
