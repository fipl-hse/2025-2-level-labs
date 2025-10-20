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
    vocabulary = build_vocabulary(remove_stop_words(clean_and_tokenize(text) or [],
                                                    stop_words) or []) or {}
    print(vocabulary)

    all_sentence_tokens = [
        token
        for sentence in sentences
        for token in remove_stop_words(clean_and_tokenize(sentence) or [], stop_words)
    ]
    error_words = find_out_of_vocab_words(all_sentence_tokens, vocabulary) or []
    print(f"\n{error_words}")

    correct_words = ['тёмной', 'московской', 'улице', 'профессор', 'квартире',
                     'думал', 'странностях', 'записывал', 'наблюдения', 'тетрадь',
                     'вечером', 'шумном', 'проспекте', 'патриса', 'лумумбы', 'собирались',
                     'прохожие', 'обсуждая', 'последние', 'московские', 'новости',
                     'деревянной', 'скамье', 'саду', 'под', 'фонарем', 'сидели', 'двое',
                     'оживлённо', 'спорили', 'судьбах', 'литературы', 'трамвай', 'заскрипел',
                     'повороте', 'остановился', 'парящего', 'моста', 'толпа', 'ждала', 'вечернего',
                     'представления', 'усталый', 'редактор', 'принёс', 'толстую', 'рукопись',
                     'контору', 'надеясь', 'наконец-то', 'найти', 'одобрение', 'коллег']
    jaccard_correct = frequency_correct = levenshtein_correct = jaro_winkler_correct = 0

    alphabet = [chr(i) for i in range(1072, 1104)]
    all_results = {}
    real_incorrect_words = 0
    for error_word in error_words:
        print(f"\nCorrection for '{error_word}':")
        jaccard_correction = find_correct_word(error_word, vocabulary,
                                               'jaccard', alphabet) or ""
        frequency_correction = find_correct_word(error_word, vocabulary,
                                                 'frequency-based', alphabet) or ""
        levenshtein_correction = find_correct_word(error_word, vocabulary,
                                                   'levenshtein', alphabet) or ""
        jaro_winkler_correction = find_correct_word(error_word, vocabulary,
                                                    'jaro-winkler', alphabet) or ""
        corrections = {
            'jaccard': jaccard_correction,
            'frequency-based': frequency_correction,
            'levenshtein': levenshtein_correction,
            'jaro-winkler': jaro_winkler_correction
        }
        has_correct = any(corr in correct_words for corr in corrections.values())
        if not has_correct:
            msg = ("The word is spelled correctly for the given context, "
            "but it is not in the vocabulary." 
                   if error_word in correct_words else
                   "The word is spelled incorrectly for the given context, "
                   "and it is not in the vocabulary.")
            print(msg)
        else:
            print("The word is spelled incorrectly. There are corrections for it.")
            real_incorrect_words += 1
            if jaccard_correction in correct_words:
                jaccard_correct += 1
            if frequency_correction in correct_words:
                frequency_correct += 1
            if levenshtein_correction in correct_words:
                levenshtein_correct += 1
            if jaro_winkler_correction in correct_words:
                jaro_winkler_correct += 1
        print(f"  Jaccard: {jaccard_correction}")
        print(f"  Frequency-based: {frequency_correction}")
        print(f"  Levenshtein: {levenshtein_correction}")
        print(f"  Jaro-Winkler: {jaro_winkler_correction}")
        all_results[error_word] = corrections
    print("\nEfficiency of methods:")
    print(f"Jaccard: {jaccard_correct}/{real_incorrect_words}")
    print(f"Frequency-based: {frequency_correct}/{real_incorrect_words}")
    print(f"Levenshtein: {levenshtein_correct}/{real_incorrect_words}")
    print(f"Jaro-Winkler: {jaro_winkler_correct}/{real_incorrect_words}")
    result = all_results
    assert result, "Result is None"


if __name__ == "__main__":
    main()
