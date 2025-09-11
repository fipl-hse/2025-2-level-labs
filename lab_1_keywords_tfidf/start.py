"""
Frequency-driven keyword extraction starter
"""

# pylint:disable=too-many-locals, unused-argument, unused-variable, invalid-name, duplicate-code
from json import load


def main() -> None:
    """
    Launches an implementation.
    """
    from main import clean_and_tokenize, remove_stop_words, calculate_frequencies, get_top_n
    with open("assets/Дюймовочка.txt", "r", encoding="utf-8") as file:
        target_text = file.read()
    tokens = clean_and_tokenize(target_text)
    print(tokens)
    with open("assets/stop_words.txt", "r", encoding="utf-8") as file:
        stop_words = file.read().split("\n")
    cleaned_tokens = remove_stop_words(clean_and_tokenize(target_text), stop_words)
    print(cleaned_tokens)
    frequencies = calculate_frequencies(cleaned_tokens)
    print(frequencies)
    top_n = get_top_n(frequencies, 5)
    print(top_n)
    with open("assets/IDF.json", "r", encoding="utf-8") as file:
        idf = load(file)
    with open("assets/corpus_frequencies.json", "r", encoding="utf-8") as file:
        corpus_freqs = load(file)
    result = top_n
    assert result, "Keywords are not extracted"


if __name__ == "__main__":
    main()
