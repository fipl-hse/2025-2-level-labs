"""
Frequency-driven keyword extraction starter
"""

# pylint:disable=too-many-locals, unused-argument, unused-variable, invalid-name, duplicate-code
from math import log
from json import load
from main import calculate_frequencies, calculate_tf, calculate_tfidf, clean_and_tokenize, get_top_n, remove_stop_words

def main() -> None:
    """
    Launches an implementation.
    """
    with open("assets/Дюймовочка.txt", "r", encoding="utf-8") as file:
        target_text = file.read()
        words=clean_and_tokenize(target_text)
    with open("assets/stop_words.txt", "r", encoding="utf-8") as file:
        stop_words = file.read().split("\n")
        final_words=remove_stop_words(words,stop_words)
    d_calc=calculate_frequencies(final_words)
    tf_dict=calculate_tf(d_calc)
    with open("assets/IDF.json", "r", encoding="utf-8") as file:
        idf = load(file)
        tfidf_dict=calculate_tfidf(tf_dict, idf)
    print(get_top_n(tfidf_dict, 10))

    with open("assets/corpus_frequencies.json", "r", encoding="utf-8") as file:
        corpus_freqs = load(file)
    result = None
    assert result, "Keywords are not extracted"


if __name__ == "__main__":
    main()
