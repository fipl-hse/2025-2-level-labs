"""
Frequency-driven keyword extraction starter
"""

# pylint:disable=too-many-locals, unused-argument, unused-variable, invalid-name, duplicate-code
from json import load    
from lab_1_keywords_tfidf.main import (
    calculate_frequencies,
    calculate_tf,
    calculate_tfidf,
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
    with open("assets/stop_words.txt", "r", encoding="utf-8") as file:
        stop_words = file.read().split("\n")
    with open("assets/IDF.json", "r", encoding="utf-8") as file:
        idf = load(file)
    with open("assets/corpus_frequencies.json", "r", encoding="utf-8") as file:
        corpus_freqs = load(file)
    result = None

    clean_and_tokenize_text = clean_and_tokenize(target_text)
    if not clean_and_tokenize:
    
        return
    
    text_without_stop_words=remove_stop_words(clean_and_tokenize_text, stop_words)
    if not text_without_stop_words:

        return
    
    calculated_frequencies=calculate_frequencies(text_without_stop_words)
    if not calculated_frequencies:

        return
    
    calculated_tf=calculate_tf(calculated_frequencies)
    if not calculated_tf:

        return
    
    calculated_tfidf=calculate_tfidf(calculated_tf, idf)
    if not calculated_tfidf:

        return
    

    top_n_words=get_top_n(calculated_tfidf, 10)

    print("Текст без стоп-слов:", text_without_stop_words)
    print("Частоты слов:", calculated_frequencies)
    print("Term Frequency для всех слов: ", calculated_tf)
    print("TF-IDF для всех слов: ", calculated_tfidf, idf)
    print("Топ-10 ключевых слов:", top_n_words)
    result=top_n_words
    assert result, "Keywords are not extracted"
    
if __name__ == "__main__":
    main()
