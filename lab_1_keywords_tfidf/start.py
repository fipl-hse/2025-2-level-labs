"""
Frequency-driven keyword extraction starter
"""

# pylint:disable=too-many-locals, unused-argument, unused-variable, invalid-name, duplicate-code
from json import load

from main import clean_and_tokenize

from main import remove_stop_words

from main import calculate_frequencies

from main import get_top_n

from main import calculate_tf


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
    unfilt_tokens = clean_and_tokenize(target_text)
    print(unfilt_tokens)
    tokens = remove_stop_words(unfilt_tokens, stop_words)
    print (tokens)
    frequencies = calculate_frequencies(tokens)
    print (frequencies)
    top_values = get_top_n(frequencies)
    print (top_values)
    term_freq = calculate_tf(frequencies)
    print (term_freq)
    result = term_freq #(изменить в 12 шаге)
    assert result, "Keywords are not extracted"


if __name__ == "__main__":
    main()
