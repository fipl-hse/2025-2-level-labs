"""
Spellcheck starter
"""

# pylint:disable=unused-variable, duplicate-code, too-many-locals

from typing import Literal

from lab_1_keywords_tfidf.main import clean_and_tokenize, remove_stop_words
from lab_2_spellcheck.main import build_vocabulary, find_correct_word, find_out_of_vocab_words


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
        
    main_tokens = clean_and_tokenize(text) or []
    filtered_main_tokens = remove_stop_words(main_tokens, stop_words) or []
    vocabulary_map = build_vocabulary(filtered_main_tokens) or {}

    all_tokens_from_sentences = []
    for sentence in sentences:
        tokens = clean_and_tokenize(sentence) or []
        filtered_tokens = remove_stop_words(tokens, stop_words) or []
        all_tokens_from_sentences.extend(filtered_tokens)

    unknown_words = find_out_of_vocab_words(all_tokens_from_sentences, vocabulary_map) or []
    print('Слова не в словаре:', unknown_words)

    rus_alphabet = list('абвгдеёжзийклмнопрстуфхцчшщъыьэюя')
    corrections_result = {}

    for unknown in unknown_words:
        print(f'\nВозможные исправления слова: {unknown}')
        word_corrections = {}
        for method_name in ['jaccard', 'frequency-based', 'levenshtein', 'jaro-winkler']:
            method: Literal['jaccard',
                            'frequency-based',
                            'levenshtein',
                            'jaro-winkler'] = method_name
            correction = find_correct_word(unknown,
                                           vocabulary_map,
                                           method_name,
                                           rus_alphabet) or None
            word_corrections[method] = correction
            print(f"  {method_name.capitalize()}: {correction}")
        corrections_result[unknown] = word_corrections

    result = corrections_result
    assert result, "Result is None"


if __name__ == "__main__":
    main()
