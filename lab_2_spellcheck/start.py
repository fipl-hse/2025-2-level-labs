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
    result = None
    cleaned_and_tokenized_text=clean_and_tokenize(text) or []

    text_without_stop_words=remove_stop_words(cleaned_and_tokenized_text, stop_words) or []

    vocabulary=build_vocabulary(text_without_stop_words) or {}

    all_elements_in_sentences=[]
    for element in sentences:
        cleaned_tokens=clean_and_tokenize(element) or []
        without_stop_words=remove_stop_words(cleaned_tokens, stop_words) or []
        all_elements_in_sentences.extend(without_stop_words)

    finded_out_of_vocab_words=find_out_of_vocab_words(all_elements_in_sentences, vocabulary) or []
    print('Слова не из словаря:', finded_out_of_vocab_words)

    russian_alphabet=list('абвгдеёжзийклмнопрстуфхцчшщъыьэюя')
    corrected_words = {}

    for token in finded_out_of_vocab_words:
        jaccard_correct=find_correct_word(token, vocabulary, 'jaccard', russian_alphabet)
        frequency_corrected=find_correct_word(
            token, vocabulary, 'frequency-based', russian_alphabet
            )
        levenshtein_corrected=find_correct_word(
            token, vocabulary, 'levenshtein', russian_alphabet
            )
        jaro_winkler_corrected=find_correct_word(
            token, vocabulary, 'jaro-winkler', russian_alphabet
            )

        corrected_words[token] = {
        'jaccard': jaccard_correct,
        'frequency-based': frequency_corrected,
        'levenshtein': levenshtein_corrected,
        'jaro-winkler': jaro_winkler_corrected
    }

    for word, corrections in corrected_words.items():
        print(f"\nСлово: '{word}'")
        print(f"Jaccard:          {corrections['jaccard']}")
        print(f"Frequency-based:  {corrections['frequency-based']}")
        print(f"Levenshtein:      {corrections['levenshtein']}")
        print(f"Jaro-Winkler:     {corrections['jaro-winkler']}")

    result=corrected_words
    print(result)
    print(f"\nИтоговый результат: {result}")
    assert result, "Result is None"

if __name__ == "__main__":
    main()
