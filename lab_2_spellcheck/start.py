"""
Spellcheck starter
"""

# pylint:disable=unused-variable, duplicate-code, too-many-locals
from lab_1_keywords_tfidf.main import clean_and_tokenize, remove_stop_words
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

    alphabet = [
        'а', 'б', 'в', 'г', 'д', 'е', 'ё', 'ж', 'з', 'и', 'й', 'к', 'л', 'м',
        'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь',
        'э', 'ю', 'я'
    ]

    cleaned_tokens = clean_and_tokenize(text)
    if not cleaned_tokens:
        return

    removed_words = remove_stop_words(cleaned_tokens, stop_words)
    if not removed_words:
        return

    built_voc = build_vocabulary(removed_words)
    if not built_voc:
        return

    results = []
    for sentence in sentences:
        sentence_tokens = clean_and_tokenize(sentence)
        sentence_filtered = (
            remove_stop_words(sentence_tokens, stop_words)
            if sentence_tokens else None
        )
        out_of_vocab_words = (
            find_out_of_vocab_words(sentence_filtered, built_voc)
            if sentence_filtered else None
        )

        if not (sentence_tokens, sentence_filtered, out_of_vocab_words):
            continue
        
        corrections = []
        for wrong_word in out_of_vocab_words:
            correct_word = (
                find_correct_word(wrong_word, built_voc, "frequency-based", alphabet) or
                find_correct_word(wrong_word, built_voc, "levenshtein", None) or
                find_correct_word(wrong_word, built_voc, "jaccard", None)
            )

            if correct_word:
                corrections.append((wrong_word, correct_word))

        if corrections:
            results.append({
                'sentence': sentence,
                'corrections': corrections
            })

    if results:
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Исходное предложение: {result['sentence']}")
            print("   Исправления:")
            for wrong, correct in result['corrections']:
                print(f"     '{wrong}' → '{correct}'")

    else:
        print("Ошибки не найдены.")

    assert results is not None
    # return final_result


if __name__ == "__main__":
    main()
