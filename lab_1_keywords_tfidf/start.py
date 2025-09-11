"""
Frequency-driven keyword extraction starter
"""

# pylint:disable=too-many-locals, unused-argument, unused-variable, invalid-name, duplicate-code
from json import load
from main import clean_and_tokenize
from main import remove_stop_words
from main import calculate_frequencies
from main import get_top_n

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
    tokenized_text = clean_and_tokenize(target_text)
    no_stopwords_text = remove_stop_words(tokenized_text, stop_words)
    frequ_dict = calculate_frequencies(no_stopwords_text)
    top_n_words = get_top_n(frequ_dict, 10)
    result = top_n_words

    print(result)
    assert result, "Keywords are not extracted"


if __name__ == "__main__":
    main()

