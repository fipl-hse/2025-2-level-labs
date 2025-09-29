"""
Frequency-driven keyword extraction starter
"""

# pylint:disable=too-many-locals, unused-argument, unused-variable, invalid-name, duplicate-code

import sys
import os
from json import load

current_dir_of_start = os.path.abspath(os.path.dirname(__file__))
path_to_project_root = os.path.abspath(os.path.join(current_dir_of_start, '..'))
sys.path.append(path_to_project_root)

from lab_1_keywords_tfidf.main import (
    calculate_frequencies,
    clean_and_tokenize,
    get_top_n,
    remove_stop_words,
)


def main() -> None:
    """
    Launches an implementation.
    """
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    file_path_thumbelina=os.path.join(current_file_dir, 'assets', 'Дюймовочка.txt')
    file_path_stop_words = os.path.join(current_file_dir, 'assets', 'stop_words.txt')
    file_path_corpus_freqs = os.path.join(current_file_dir, 'assets', 'corpus_frequencies.json')

    try:
        with open(file_path_thumbelina, "r", encoding="utf-8") as file:
            target_text = file.read()
        tokens = clean_and_tokenize(target_text)
        with open(file_path_stop_words, "r", encoding="utf-8") as file:
            stop_words = file.read().split("\n")
        if tokens is not None:
            cleaned_tokens = remove_stop_words(tokens, stop_words)
        if cleaned_tokens is not None:
            frequencies = calculate_frequencies(cleaned_tokens)
        with open(file_path_corpus_freqs, "r", encoding="utf-8") as file:
            corpus_freqs = load(file)
        if frequencies is not None:
            result = get_top_n(frequencies, 10)
    except FileNotFoundError as e:
        print(f"Error: File not found: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return
    if result is not None:
        print(result)
    assert result is not None, "Keywords are not extracted"

if __name__ == "__main__":
    main()