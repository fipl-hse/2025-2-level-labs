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
        with open("assets/stop_words.txt", "r", encoding="utf-8") as file:
            stop_words = file.read().split("\n")
        cleaned_tokens = remove_stop_words(tokens, stop_words)
        freq_dict = calculate_frequencies(cleaned_tokens)
        with open("assets/IDF.json", "r", encoding="utf-8") as file:
            idf = json.load(file)
        with open("assets/corpus_frequencies.json", "r", encoding="utf-8") as file:
            corpus_freqs = json.load(file)
        tf = calculate_tf(freq_dict)
        tfidf = calculate_tfidf(tf, idf)
        expected_freqs = calculate_expected_frequency(freq_dict, corpus_freqs)
        chi_vals = calculate_chi_values(expected_freqs, freq_dict)
        significant_words = extract_significant_words(chi_vals, 0.05)
        top_chi = get_top_n(significant_words, 10) if significant_words else []
        all_steps_successful = (
            tokens is not None and
            cleaned_tokens is not None and
            freq_dict is not None and
            tf is not None and
            tfidf is not None and
            expected_freqs is not None and
            chi_vals is not None and
            significant_words is not None and
            top_chi is not None
        )
        if all_steps_successful:
            print("Топ-10 ключевых слов по хи-квадрат:", top_chi)
            assert top_chi == ['дюймовочка', 'ласточка', 'крот', 'дюймовочке', 
                             'дюймовочку', 'мышь', 'норки', 'прощай', 
                             'дюймовочки', 'лист'], "Результат не соответствует ожидаемому"
        else:
            print("Ошибка в процессе обработки текста")
    except FileNotFoundError as e:
        print(f"Файл не найден: {e}")

if __name__ == "__main__":
    main()
