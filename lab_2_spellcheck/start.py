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
    calculate_distance,
    calculate_levenshtein_distance,
    fill_levenshtein_matrix,
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
    russian_alphabet = [
        'а', 'б', 'в', 'г', 'д', 'е', 'ё', 'ж', 'з', 'и', 'й', 'к', 'л', 'м',
        'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ',
        'ы', 'ь', 'э', 'ю', 'я'
    ]
    tokens = clean_and_tokenize(text)
    tokens_without_stopwords = remove_stop_words(tokens, stop_words) if tokens else None
    vocabulary = build_vocabulary(tokens_without_stopwords) if tokens_without_stopwords else None
    if tokens is None or tokens_without_stopwords is None or vocabulary is None:
        return
    if not sentences:
        return
    sentence = sentences[0]
    print("\nSentence 1")
    print(f"Original: {sentence}")
    sentence_tokens = clean_and_tokenize(sentence)
    sentence_tokens_without_stopwords = remove_stop_words(sentence_tokens, stop_words) if sentence_tokens else None
    oov_words = find_out_of_vocab_words(sentence_tokens_without_stopwords, vocabulary) if sentence_tokens_without_stopwords else None
    if not sentences or sentence_tokens is None or sentence_tokens_without_stopwords is None or oov_words is None:
        return
    print(f"Out-of-vocabulary words: {oov_words}")
    for wrong_word in oov_words:
        print(f"\nProcessing word: '{wrong_word}'")
        for method in ("jaccard", "frequency-based", "levenshtein", "jaro-winkler"):
            distances = calculate_distance(wrong_word, vocabulary, method, russian_alphabet)
            correction = find_correct_word(wrong_word, vocabulary, method, russian_alphabet)
            if distances:
                top_candidates = sorted(distances.items(), key=lambda x: x[1])[:3]
                print(f"{method}: '{wrong_word}' -> '{correction}'")
                print(f"Top 3 candidates: {top_candidates}")
                if method == "levenshtein" and correction and (filled_matrix := fill_levenshtein_matrix(wrong_word, correction)):
                    print(f"Levenshtein matrix for '{wrong_word}' and '{correction}':")
                    for row in filled_matrix:
                        print(f"      {row}")
                    distance = calculate_levenshtein_distance(wrong_word, correction)
                    print(f"Final distance: {distance}")
            else:
                print(f"{method}: Failed to calculate distances")
    result = distance
    assert result, "Result is None"


if __name__ == "__main__":
    main()
