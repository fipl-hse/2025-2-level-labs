"""
Frequency-driven keyword extraction starter
"""

# pylint:disable=too-many-locals, unused-argument, unused-variable, invalid-name, duplicate-code
from json import load

from main import (
    calculate_chi_values,
    calculate_expected_frequency,
    calculate_frequencies,
    clean_and_tokenize,
    extract_significant_words,
    get_top_n,
    remove_stop_words,
)

FUNCTIONS: tuple[callable] = (
    clean_and_tokenize,
    remove_stop_words,
    calculate_frequencies,
    calculate_expected_frequency,
    calculate_chi_values,
    extract_significant_words,
    get_top_n
)


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
    current_result = target_text

    for function in FUNCTIONS:
        if function.__code__.co_argcount == 1:
            current_result = function(current_result)

        elif function.__name__ == "remove_stop_words":
            result = function(current_result, stop_words)

        elif function.__name__ == "calculate_expected_frequency":
            current_result = function(current_result, corpus_freqs)

        elif function.__name__ == "calculate_chi_values":
            current_result = function(current_result, calculated_frequencies)


        if not current_result:
            break


        if function.__name__ == "calculate_frequencies":
            calculated_frequencies = current_result

        if function.__name__ == "get_top_n":
            result = function(current_result, 10)

    result = current_result
    assert result, "Keywords are not extracted"


if __name__ == "__main__":
    main()
