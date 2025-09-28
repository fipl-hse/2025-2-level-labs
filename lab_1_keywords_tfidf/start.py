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
    

    # Шаг 2: токенизация
    tokens = clean_and_tokenize(target_text)

    # Шаг 3: убираем стоп-слова
    tokens_no_stop = remove_stop_words(tokens, stop_words)

    # Шаг 4: частоты
    freqs = calculate_frequencies(tokens_no_stop)

    # Шаг 5: топ-10 по частоте
    top_freq = get_top_n(freqs, 10)

    # Шаг 6: TF
    tf_vals = calculate_tf(freqs)

    # Шаг 7: TF-IDF
    tfidf_vals = calculate_tfidf(tf_vals, idf)
    top_tfidf = get_top_n(tfidf_vals, 10)

    print("Топ-10 слов по частоте:", top_freq)
    print("Топ-10 слов по TF-IDF:", top_tfidf)


if __name__ == "__main__":
    main()
