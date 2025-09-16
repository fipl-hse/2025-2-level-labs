"""
Frequency-driven keyword extraction starter
"""

# pylint:disable=too-many-locals, unused-argument, unused-variable, invalid-name, duplicate-code
from json import load

from lab_1_keywords_tfidf.main import (calculate_frequencies, clean_and_tokenize, get_top_n,
                                        remove_stop_words, calculate_tf, calculate_tfidf)


def main() -> None:
    """
    Launches an implementation.
    """
    with open("assets/Дюймовочка.txt", "r", encoding="utf-8") as file:
        target_text = file.read()
    tokens = clean_and_tokenize(target_text)
    #print(tokens)
    with open("assets/stop_words.txt", "r", encoding="utf-8") as file:
        stop_words = file.read().split("\n")
    cleaned_tokens = remove_stop_words(clean_and_tokenize(target_text), stop_words)
    #print(cleaned_tokens)
    frequencies = calculate_frequencies(cleaned_tokens)
    #print(frequencies)
    top_n = get_top_n(frequencies, 5)
    #print(top_n)
    tf_d = calculate_tf(frequencies)
    #print(tf_d)
    with open("assets/IDF.json", "r", encoding="utf-8") as file:
        idf = load(file)
    idf_d = calculate_tfidf(tf_d, idf)
    #print(idf_d)
    top_idf = get_top_n(idf_d, 10)
    print(top_idf)
    with open("assets/corpus_frequencies.json", "r", encoding="utf-8") as file:
        corpus_freqs = load(file)
    result = idf_d
    assert result, "Keywords are not extracted"


if __name__ == "__main__":
    main()
