"""
Spellcheck starter
"""

# pylint:disable=unused-variable, duplicate-code, too-many-locals
from lab_1_keywords_tfidf.main import (
    clean_and_tokenize,
    remove_stop_words,
)
from lab_2_spellcheck.main import (
    build_vocabulary,
    find_correct_word,
    find_out_of_vocab_words,
)


def main() -> None:
    """
    Launches an implementation.
    """
    with open("assets/Master_and_Margarita_chapter1.txt", "r", encoding="utf-8") as file:
        text = file.read()
    with open("assets/stop_words.txt", "r", encoding="utf-8") as file:
        stop_words = file.read().split("\n")
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
    vocabulary = build_vocabulary(tokens_without_stopwords) or {}
    all_sentence_tokens = []
    for sentence in sentences:
        sentence_tokens = clean_and_tokenize(sentence) or []
        sentence_tokens_without_stop_words = remove_stop_words(sentence_tokens, stop_words) or []
        all_sentence_tokens.extend(sentence_tokens_without_stop_words)
    error_words = find_out_of_vocab_words(all_sentence_tokens, vocabulary) or []
    print(error_words)
    alphabet = [chr(i) for i in range(1072, 1104)]
    all_results = {}
    for error_word in error_words:
        print(f"\nCorrection for '{error_word}':")
        jaccard_correction = find_correct_word(error_word, vocabulary,
                                               'jaccard', alphabet) or {}
        frequency_correction = find_correct_word(error_word, vocabulary,
                                                 'frequency-based', alphabet) or {}
        levenshtein_correction = find_correct_word(error_word, vocabulary,
                                                   'levenshtein', alphabet) or {}
        jaro_winkler_correction = find_correct_word(error_word, vocabulary,
                                                    'jaro-winkler', alphabet) or {}
        print(f"  Jaccard: {jaccard_correction}")
        print(f"  Frequency-based: {frequency_correction}")
        print(f"  Levenshtein: {levenshtein_correction}")
        print(f"  Jaro-Winkler: {jaro_winkler_correction}")
        all_results[error_word] = {
            'jaccard': jaccard_correction,
            'frequency-based': frequency_correction,
            'levenshtein': levenshtein_correction,
            'jaro-winkler': jaro_winkler_correction
        }
    right_corrections = ['тёмной', 'московской', 'улице', 'профессор', 'квартире',
                     'думал', 'странностях', 'записывал', 'наблюдения', 'тетрадь',
                     'вечером', 'шумном', 'проспекте', 'патриса', 'лумумбы', 'собирались',
                     'прохожие', 'обсуждая', 'последние', 'московские', 'новости',
                     'деревянной', 'скамье', 'саду', 'под', 'фонарём', 'сидели', 'двое',
                     'оживлённо', 'спорили', 'судьбах', 'литературы', 'трамвай', 'заскрипел',
                     'повороте', 'остановился', 'парящего', 'моста', 'толпа', 'ждала', 'вечернего',
                     'представления', 'усталый', 'редактор', 'принёс', 'толстую', 'рукопись',
                     'контору', 'надеясь', 'наконец-то', 'найти', 'одобрение', 'коллег']
    scores = {
    'jaccard': 0,
    'frequency-based': 0,
    'levenshtein': 0,
    'jaro-winkler': 0
    }
    total_evaluated_errors = 0
    for i, error_word in enumerate(error_words):
        if i < len(right_corrections) and right_corrections[i] is not None:
            correct_word = right_corrections[i]
            total_evaluated_errors += 1
            corrections = all_results[error_word]
            for method, suggested_correction in corrections.items():
                if isinstance(suggested_correction, dict) and 'word' in suggested_correction:
                    if suggested_correction['word'] == correct_word:
                        scores[method] += 1
                elif suggested_correction == correct_word:
                    scores[method] += 1       
    total_evaluated_errors = min(len(error_words), len(right_corrections))
    print(f"ошибок для оценки: {total_evaluated_errors}")
    for method, score in scores.items():
        print(f"{method} | {score}/{total_evaluated_errors}")   
    if total_evaluated_errors > 0:
        best_method, best_score = max(scores.items(), key=lambda x: x[1])
        print(f"\nЛучший метод: {best_method} ({best_score}/{total_evaluated_errors})")
    result = all_results
    assert result, "Result is None"


if __name__ == "__main__":
    main()
