"""
Frequency-driven keyword extraction starter
"""

import sys
import os
from json import load

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from main import (
        calculate_frequencies,
        calculate_tf,
        calculate_tfidf,
        clean_and_tokenize,
        get_top_n,
        remove_stop_words,
    )
except ImportError:
    from .main import (
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

    tokens = clean_and_tokenize(target_text)
    if not tokens:
        print("Ошибка: не удалось токенизировать текст")
        return

    filtered_tokens = remove_stop_words(tokens, stop_words)
    if not filtered_tokens:
        print("Ошибка: не удалось удалить стоп-слова")
        return

    frequencies = calculate_frequencies(filtered_tokens)
    if not frequencies:
        print("Ошибка: не удалось вычислить частоты")
        return

    tf_scores = calculate_tf(frequencies)
    if not tf_scores:
        print("Ошибка: не удалось вычислить TF")
        return

    tfidf_scores = calculate_tfidf(tf_scores, idf)
    if not tfidf_scores:
        print("Ошибка: не удалось вычислить TF-IDF")
        return

    result = get_top_n(tfidf_scores, 10)

    if not result:
        print("Ошибка: не удалось извлечь ключевые слова")
        return

    print("Извлеченные ключевые слова:")
    for i, keyword in enumerate(result, 1):
        tfidf_score = tfidf_scores.get(keyword, 0.0)
        print(f"{i}. {keyword} (TF-IDF: {tfidf_score:.4f})")

    assert result, "Keywords are not extracted"


if __name__ == "__main__":
    main()