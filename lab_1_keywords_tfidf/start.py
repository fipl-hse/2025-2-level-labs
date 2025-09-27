"""
Frequency-driven keyword extraction starter
"""

# pylint:disable=too-many-locals, unused-argument, unused-variable, invalid-name, duplicate-code
from json import load

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
    with open("assets/corpus_frequencies.json", "r", encoding="utf-8") as file:
        corpus_freqs = load(file)
    result = None
    cleaned_text = clean_and_tokenize(target_text)
    if not cleaned_text:
        return
    cleaned_text = remove_stop_words(cleaned_text, stop_words)
    if not cleaned_text:
        return
    text_frequencies = calculate_frequencies(cleaned_text)
    if not text_frequencies:
        return
    if not only_key_words:
        return
    top_words = get_top_n(only_key_words, 10)
    result = top_words
    assert result, "Keywords are not extracted"
    


