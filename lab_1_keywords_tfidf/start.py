"""
Frequency-driven keyword extraction starter
"""

# pylint:disable=too-many-locals, unused-argument, unused-variable, invalid-name, duplicate-code
from json import load
from lab_1_keywords_tfidf.main import (
    clean_and_tokenize,
    remove_stop_words,
    calculate_frequencies,
    get_top_n,
    calculate_tf,
    calculate_tfidf,
    calculate_expected_frequency
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
    assert result, "Keywords are not extracted"

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

    top_frequency_keywords = get_top_n(frequencies, 10)
    print("=== МЕТОД 1: Частотный анализ ===")
    print("Топ-10 ключевых слов по частоте:")
    for i, keyword in enumerate(top_frequency_keywords, 1):
        print(f"{i}. {keyword} (частота: {frequencies[keyword]})")
    print()

    tf_scores = calculate_tf(frequencies)
    if not tf_scores:
        print("Ошибка: не удалось вычислить TF")
        return

    tfidf_scores = calculate_tfidf(tf_scores, idf)
    if not tfidf_scores:
        print("Ошибка: не удалось вычислить TF-IDF")
        return

    top_tfidf_keywords = get_top_n(tfidf_scores, 10)
    print("=== МЕТОД 2: TF-IDF анализ ===")
    print("Топ-10 ключевых слов по TF-IDF:")
    for i, keyword in enumerate(top_tfidf_keywords, 1):
        print(f"{i}. {keyword} (TF-IDF: {tfidf_scores[keyword]:.4f})")
    print()

    expected_freqs = calculate_expected_frequency(frequencies, corpus_freqs)
    if expected_freqs:
        top_expected_keywords = get_top_n(expected_freqs, 10)
        print("=== МЕТОД 3: Ожидаемые частоты ===")
        print("Топ-10 ключевых слов по ожидаемым частотам:")
        for i, keyword in enumerate(top_expected_keywords, 1):
            print(f"{i}. {keyword} (ожидаемая частота: {expected_freqs[keyword]})")
    else:
        print(" Выявлено, что метод ожидаемых частот не применим для данного текста")
    print()

    result = top_tfidf_keywords
    print(f"Итоговый результат (топ TF-IDF ключевых слов): {result}")
    assert result, "Keywords are not extracted"


if __name__ == "__main__":
    main()
