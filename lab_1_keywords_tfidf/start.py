"""
Frequency-driven keyword extraction starter
"""

import json

from lab_1_keywords_tfidf.main import (
    calculate_chi_values,
    calculate_expected_frequency,
    calculate_frequencies,
    calculate_tf,
    calculate_tfidf,
    clean_and_tokenize,
    extract_significant_words,
    get_top_n,
    remove_stop_words,
)


def main() -> None:
    """
    Launches an implementation.
    """
    try:
        with open("assets/Дюймовочка.txt", "r", encoding="utf-8") as file:
            target_text = file.read()
        tokens = clean_and_tokenize(target_text)
        if not tokens:
            print("Ошибка на этапе предобработки текста")
            return
        with open("assets/stop_words.txt", "r", encoding="utf-8") as file:
            stop_words = file.read().split("\n")
        cleaned_tokens = remove_stop_words(tokens, stop_words)
        if not cleaned_tokens:
            print("Ошибка на этапе удаления стоп-слов")
            return
        freq_dict = calculate_frequencies(cleaned_tokens)
        if not freq_dict:
            print("Ошибка на этапе предобработки текста")
            return
        with open("assets/IDF.json", "r", encoding="utf-8") as file:
            idf = json.load(file)
        with open("assets/corpus_frequencies.json", "r", encoding="utf-8") as file:
            corpus_freqs = json.load(file)
        tf = calculate_tf(freq_dict)
        if not tf:
            print("Ошибка в расчетах TF")
            return
        tfidf = calculate_tfidf(tf, idf)
        if not tfidf:
            print("Ошибка в расчетах TF-IDF")
            return
        expected_freqs = calculate_expected_frequency(freq_dict, corpus_freqs)
        if not expected_freqs:
            print("Ошибка в расчетах ожидаемых частот")
            return
        chi_vals = calculate_chi_values(expected_freqs, freq_dict)
        if not chi_vals:
            print("Ошибка в расчетах статистик")
            return
        significant_words = extract_significant_words(chi_vals, 0.05)
        top_chi = get_top_n(significant_words, 10) if significant_words else []
        print("Топ-10 ключевых слов по хи-квадрат:", top_chi)
    except FileNotFoundError as e:
        print(f"Файл не найден: {e}")

if __name__ == "__main__":
    main()
