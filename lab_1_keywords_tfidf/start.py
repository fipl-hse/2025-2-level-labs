"""
Frequency-driven keyword extraction starter
"""

# pylint:disable=too-many-locals, unused-argument, unused-variable, invalid-name, duplicate-code
from json import load

'''
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
    assert result, "Keywords are not extracted"


if __name__ == "__main__":
    main()
'''

from typing import Any


def check_list(user_input: Any, elements_type: type, can_be_empty: bool) -> bool:
    if user_input == []:
        check_result = can_be_empty == True
    elif type(user_input) == list:
        for element in user_input:
            if type(element) != elements_type:
                check_result = False
                break
            else:
                check_result = True
    else:
        check_result = False
    return check_result


print(check_list([5], int, False))