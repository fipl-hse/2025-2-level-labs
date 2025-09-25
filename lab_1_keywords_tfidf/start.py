"""
Frequency-driven keyword extraction starter
"""
from json import load

from lab_1_keywords_tfidf.main import calculate_chi_values, calculate_expected_frequency, calculate_frequencies, calculate_tf, calculate_tfidf, clean_and_tokenize, extract_significant_words, get_top_n, remove_stop_words


# pylint:disable=too-many-locals, unused-argument, unused-variable, invalid-name, duplicate-code
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
    cleaned_text = clean_and_tokenize(target_text)
    if not cleaned_text:
        return
    cleaned_text = remove_stop_words(cleaned_text, stop_words)
    if not cleaned_text:
        return
    text_frequencies = calculate_frequencies(cleaned_text)
    if not text_frequencies:
        return
    tf_frequencies = calculate_tf(text_frequencies)
    if not tf_frequencies:
        return
    tfidf_frequencies = calculate_tfidf(tf_frequencies, idf)
    if not tfidf_frequencies:
        return
    expected_frequencies = calculate_expected_frequency(text_frequencies, corpus_freqs)
    chi_value_frequency = calculate_chi_values(tfidf_frequencies, corpus_freqs)
    only_key_words = extract_significant_words(tfidf_frequencies, 0.001)
    if not only_key_words:
        return
    top_words = get_top_n(only_key_words, 10)
    result = top_words
    assert result, "Keywords are not extracted"


if __name__ == "__main__":
    main()
