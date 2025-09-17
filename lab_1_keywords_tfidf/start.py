"""
Frequency-driven keyword extraction starter
"""

# pylint:disable=too-many-locals, unused-argument, unused-variable, invalid-name, duplicate-code
from json import load
from lab_1_keywords_tfidf.main import(remove_stop_words, clean_and_tokenize, calculate_frequencies, get_top_n)

def main() -> None:
    """
    Launches an implementation.
    """
    with open("lab_1_keywords_tfidf/assets/Дюймовочка.txt", "r", encoding="utf-8") as file:
        target_text = file.read()
    with open("lab_1_keywords_tfidf/assets/stop_words.txt", "r", encoding="utf-8") as file:
        stop_words = file.read().split("\n")
    with open("lab_1_keywords_tfidf/assets/IDF.json", "r", encoding="utf-8") as file:
        idf = load(file)
    with open("lab_1_keywords_tfidf/assets/corpus_frequencies.json", "r", encoding="utf-8") as file:
        corpus_freqs = load(file)
    result = None
    assert result, "Keywords are not extracted"

    if __name__ == "__main__":
        main()
    

    clean_and_tokenize_text = clean_and_tokenize(target_text)
    print("Cleaned_text:", clean_and_tokenize_text)

    text_without_stop_words=remove_stop_words(clean_and_tokenize_text, stop_words)
    print("Text without stop words: ", text_without_stop_words)

    calculated_frequencies=calculate_frequencies(text_without_stop_words)

    top_n_text=get_top_n(calculated_frequencies, 10)

    result=top_n_text


