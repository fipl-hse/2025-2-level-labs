"""
Frequency-driven keyword extraction starter
"""

# pylint:disable=too-many-locals, unused-argument, unused-variable, invalid-name, duplicate-code
from json import load
from lab_1_keywords_tfidf.main import clean_and_tokenize, get_top_n, calculate_frequencies, calculate_expected_frequency, calculate_tf, calculate_tfidf, remove_stop_words, extract_significant_words

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
    # result = None
    # assert result, "Keywords are not extracted"
    cleaned_text = clean_and_tokenize(target_text)
    cleaned_text = remove_stop_words(cleaned_text, stop_words)
    text_frequencies = calculate_frequencies(cleaned_text)
    tf_frequencies = calculate_tf(text_frequencies)
    tfidf_frequencies = calculate_tfidf(tf_frequencies, idf)
    expected_frequencies = calculate_expected_frequency(text_frequencies, corpus_freqs)
    only_key_words = extract_significant_words(tfidf_frequencies)
    top_words = get_top_n(only_key_words, 10)
    result = top_words
    assert result, "Keywords are not extracted"


if __name__ == "__main__":
    main()
