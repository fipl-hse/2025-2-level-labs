"""
Frequency-driven keyword extraction starter
"""

<<<<<<< HEAD
import os
import sys
from json import load

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from lab_1_keywords_tfidf.main import clean_and_tokenize, remove_stop_words
    from lab_2_spellcheck.main import (
        build_vocabulary,
        calculate_distance,
        calculate_frequency_distance,
        calculate_jaccard_distance,
        calculate_jaro_winkler_distance,
        calculate_levenshtein_distance,
        find_correct_word,
        find_out_of_vocab_words,
    )
except ImportError:
    from lab_1_keywords_tfidf.main import clean_and_tokenize, remove_stop_words
    from lab_2_spellcheck.main import (
        build_vocabulary,
        calculate_distance,
        calculate_frequency_distance,
        calculate_jaccard_distance,
        calculate_jaro_winkler_distance,
        calculate_levenshtein_distance,
        find_correct_word,
        find_out_of_vocab_words,
    )
=======
# pylint:disable=unused-variable, duplicate-code, too-many-locals
from lab_1_keywords_tfidf.main import clean_and_tokenize, remove_stop_words
from lab_2_spellcheck.main import (
    build_vocabulary,
    calculate_distance,
    calculate_frequency_distance,
    calculate_jaro_winkler_distance,
    calculate_levenshtein_distance,
    find_correct_word,
    find_out_of_vocab_words,
)
>>>>>>> upstream/main


def main() -> None:
    with open("assets/Master_and_Margarita_chapter1.txt", "r", encoding="utf-8") as file:
        text = file.read()

    with open("assets/stop_words.txt", "r", encoding="utf-8") as file:
        stop_words = file.read().split("\n")
<<<<<<< HEAD

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

    print("\n" + "=" * 50)
    print("СРАВНЕНИЕ МЕТРИК ДЛЯ ИСПРАВЛЕНИЯ ОПЕЧАТОК")
    print("=" * 50)

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

    print("\n" + "=" * 50)
    print("ДЕМОНСТРАЦИЯ calculate_distance ДЛЯ РАЗНЫХ МЕТОДОВ")
    print("=" * 50)

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

    result = True
    assert result, "Spellcheck demonstration completed successfully"
=======
    with (
        open("assets/incorrect_sentence_1.txt", "r", encoding="utf-8") as f1,
        open("assets/incorrect_sentence_2.txt", "r", encoding="utf-8") as f2,
        open("assets/incorrect_sentence_3.txt", "r", encoding="utf-8") as f3,
        open("assets/incorrect_sentence_4.txt", "r", encoding="utf-8") as f4,
        open("assets/incorrect_sentence_5.txt", "r", encoding="utf-8") as f5,
    ):
        sentences = [f.read() for f in (f1, f2, f3, f4, f5)]
    tokens = clean_and_tokenize(text) or []
    tokens_without_stopwords = remove_stop_words(tokens, stop_words) or []
    tokens_vocab = build_vocabulary(tokens_without_stopwords) or {}
    print(tokens_vocab)

    tokens_not_in_vocab = find_out_of_vocab_words(tokens_without_stopwords, tokens_vocab) or []
    print(tokens_not_in_vocab)

    jaccard_distance = calculate_distance("кот", {"кот": 0.5, "пёс": 0.5},
                                                 method = "jaccard") or {}
    print(jaccard_distance)

    alphabet = list("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")
    freq_distances = calculate_frequency_distance("маладой", tokens_vocab, alphabet) or {}
    print(freq_distances)

    levenshtein_distance = calculate_levenshtein_distance("кот", "кто")
    print(levenshtein_distance)

    jaro_winkler_distance = calculate_jaro_winkler_distance("кот", "кто")
    print(jaro_winkler_distance)
    result = jaro_winkler_distance

    all_wrong_words = []
    for sentence in sentences:
        sentence_tokens = clean_and_tokenize(sentence) or []
        out_of_vocab = find_out_of_vocab_words(sentence_tokens, tokens_vocab) or []
        all_wrong_words.extend(out_of_vocab)
    unique_wrong_words = sorted(set(all_wrong_words))

    for wrong_word in unique_wrong_words:
        print(f"Исправления для слова '{wrong_word}':")
        correct_word = find_correct_word(wrong_word, tokens_vocab, "jaccard", alphabet)
        if correct_word and correct_word != wrong_word:
            print(f"jaccard: {correct_word}")
        correct_word = find_correct_word(wrong_word, tokens_vocab, "frequency-based", alphabet)
        if correct_word and correct_word != wrong_word:
            print(f"frequency-based: {correct_word}")
        correct_word = find_correct_word(wrong_word, tokens_vocab, "levenshtein", alphabet)
        if correct_word and correct_word != wrong_word:
            print(f"levenshtein: {correct_word}")
        correct_word = find_correct_word(wrong_word, tokens_vocab, "jaro-winkler", alphabet)
        if correct_word and correct_word != wrong_word:
            print(f"jaro-winkler: {correct_word}")
    assert result, "Result is None"
>>>>>>> upstream/main


if __name__ == "__main__":
    main()
