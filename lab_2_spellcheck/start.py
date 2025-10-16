"""
Frequency-driven keyword extraction starter
"""

import sys
import os
from json import load

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from lab_1_keywords_tfidf.main import clean_and_tokenize, remove_stop_words
    from lab_2_spellcheck.main import (
        build_vocabulary,
        find_out_of_vocab_words,
        calculate_distance,
        find_correct_word,
        calculate_jaccard_distance,
        calculate_levenshtein_distance,
        calculate_jaro_winkler_distance,
        calculate_frequency_distance
    )
except ImportError:
    from lab_1_keywords_tfidf.main import clean_and_tokenize, remove_stop_words
    from lab_2_spellcheck.main import (
        build_vocabulary,
        find_out_of_vocab_words,
        calculate_distance,
        find_correct_word,
        calculate_jaccard_distance,
        calculate_levenshtein_distance,
        calculate_jaro_winkler_distance,
        calculate_frequency_distance
    )


def main() -> None:
    with open("assets/Master_and_Margarita_chapter1.txt", "r", encoding="utf-8") as file:
        text = file.read()
    
    with open("assets/stop_words.txt", "r", encoding="utf-8") as file:
        stop_words = file.read().split("\n")
    
    tokens = clean_and_tokenize(text)
    if not tokens:
        print("Ошибка: не удалось токенизировать текст")
        return
    
    filtered_tokens = remove_stop_words(tokens, stop_words)
    if not filtered_tokens:
        print("Ошибка: не удалось удалить стоп-слова")
        return
    
    print(f"Токенизировано слов: {len(filtered_tokens)}")
    
    vocabulary = build_vocabulary(filtered_tokens)
    if not vocabulary:
        print("Ошибка: не удалось построить словарь")
        return
    
    print(f"\nСловарь содержит {len(vocabulary)} уникальных слов")
    print("Первые 10 слов словаря:")
    for i, (word, freq) in enumerate(list(vocabulary.items())[:10]):
        print(f"{i+1}. {word}: {freq:.4f}")
    
    out_of_vocab = find_out_of_vocab_words(filtered_tokens, vocabulary)
    if out_of_vocab is None:
        print("Ошибка: не удалось найти слова вне словаря")
        return
    
    print(f"\nСлов вне словаря: {len(out_of_vocab)}")
    if out_of_vocab:
        print("Примеры слов вне словаря:")
        for word in list(set(out_of_vocab))[:10]:
            print(f"- {word}")
    
    test_words = ["кот", "малоко", "молако", "превет", "динь"]
    alphabet_ru = list("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")
    
    print("\n" + "="*50)
    print("СРАВНЕНИЕ МЕТРИК ДЛЯ ИСПРАВЛЕНИЯ ОПЕЧАТОК")
    print("="*50)
    
    for wrong_word in test_words:
        print(f"\nСлово с опечаткой: '{wrong_word}'")
        print("-" * 30)
        
        jaccard_result = find_correct_word(wrong_word, vocabulary, "jaccard")
        if jaccard_result:
            jaccard_dist = calculate_jaccard_distance(wrong_word, jaccard_result)
            print(f"Жаккар: '{jaccard_result}' (расстояние: {jaccard_dist:.4f})")
        
        freq_result = find_correct_word(wrong_word, vocabulary, "frequency-based", alphabet_ru)
        if freq_result:
            freq_distances = calculate_frequency_distance(wrong_word, vocabulary, alphabet_ru)
            if freq_distances and freq_result in freq_distances:
                freq_dist = freq_distances[freq_result]
                print(f"Частотность: '{freq_result}' (расстояние: {freq_dist:.4f})")
        
        levenshtein_result = find_correct_word(wrong_word, vocabulary, "levenshtein")
        if levenshtein_result:
            levenshtein_dist = calculate_levenshtein_distance(wrong_word, levenshtein_result)
            print(f"Левенштейн: '{levenshtein_result}' (расстояние: {levenshtein_dist})")
        
        jaro_winkler_result = find_correct_word(wrong_word, vocabulary, "jaro-winkler")
        if jaro_winkler_result:
            jaro_winkler_dist = calculate_jaro_winkler_distance(wrong_word, jaro_winkler_result)
            print(f"Джаро-Винклер: '{jaro_winkler_result}' (расстояние: {jaro_winkler_dist:.4f})")
    
    test_word = "малоко"
    sample_vocab = {"молоко": 0.1, "маленький": 0.05, "молоток": 0.03, "малина": 0.02}
    
    print("\n" + "="*50)
    print("ДЕМОНСТРАЦИЯ calculate_distance ДЛЯ РАЗНЫХ МЕТОДОВ")
    print("="*50)
    
    print(f"\nСлово для проверки: '{test_word}'")
    print("Пример словаря:", sample_vocab)
    
    jaccard_distances = calculate_distance(test_word, sample_vocab, "jaccard")
    if jaccard_distances:
        print(f"\nРасстояния Жаккара:")
        for word, dist in jaccard_distances.items():
            print(f"  {word}: {dist:.4f}")
    
    freq_distances = calculate_distance(test_word, sample_vocab, "frequency-based", alphabet_ru)
    if freq_distances:
        print(f"\nРасстояния на основе частотности:")
        for word, dist in freq_distances.items():
            print(f"  {word}: {dist:.4f}")
    
    levenshtein_distances = calculate_distance(test_word, sample_vocab, "levenshtein")
    if levenshtein_distances:
        print(f"\nРасстояния Левенштейна:")
        for word, dist in levenshtein_distances.items():
            print(f"  {word}: {dist}")
    
    jaro_winkler_distances = calculate_distance(test_word, sample_vocab, "jaro-winkler")
    if jaro_winkler_distances:
        print(f"\nРасстояния Джаро-Винклера:")
        for word, dist in jaro_winkler_distances.items():
            print(f"  {word}: {dist:.4f}")


if __name__ == "__main__":
    main()