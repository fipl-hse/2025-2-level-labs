"""
Frequency-driven keyword extraction starter
"""

# pylint:disable=too-many-locals, unused-argument, unused-variable, invalid-name, duplicate-code
from json import load

import sys
import os
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
project_root_dir = os.path.abspath(os.path.join(current_dir, '..'))
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)

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
    with open("assets/Дюймовочка.txt", "r", encoding="utf-8") as file:
        target_text = file.read()
    tokens = clean_and_tokenize(target_text)
    with open("assets/stop_words.txt", "r", encoding="utf-8") as file:
        stop_words = file.read().split("\n")
    if not tokens:
        return remove_stop_words(tokens, stop_words)
    if not cleaned_tokens:
        return calculate_frequencies(cleaned_tokens)
    with open("assets/corpus_frequencies.json", "r", encoding="utf-8") as file:
        corpus_freqs = load(file)
    if not result:
        print(get_top_n(result, 10))
    assert result, "Keywords are not extracted"



