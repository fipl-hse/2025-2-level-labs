"""
Frequency-driven keyword extraction starter
"""

# pylint:disable=too-many-locals, unused-argument, unused-variable, invalid-name, duplicate-code
import os
from json import load
from main import clean_and_tokenize
from main import check_dict
from main import check_list
from main import remove_stop_words
from main import calculate_frequencies
from main import get_top_n

BASE_DIR = os.path.dirname(__file__)
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

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

    result_text = clean_and_tokenize(target_text)
    result_text = remove_stop_words(result_text, stop_words)
    result_text = calculate_frequencies(result_text)
    result = result_text
    assert result, "Keywords are not extracted" 




if __name__ == "__main__":
    main()
