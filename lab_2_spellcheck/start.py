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
    if tokens is None:
        return
    tokens_without_stopwords = remove_stop_words(tokens, stop_words)
    if tokens_without_stopwords is None:
        return
    vocabulary = build_vocabulary(tokens_without_stopwords)
    if vocabulary is None:
        return
    print("Top-5 words with relative frequency:")
    for word, freq in sorted(vocabulary.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"{word}: {freq:.4f}")
    for i, sentence in enumerate(sentences[:2], 1):
        print(f"\nSentence {i}")
        print(f"Original: {sentence}")
        sentence_tokens = clean_and_tokenize(sentence)
        if sentence_tokens is None:
            continue
        sentence_tokens_without_stopwords = remove_stop_words(sentence_tokens, stop_words)
        if sentence_tokens_without_stopwords is None:
            continue
        oov_words = find_out_of_vocab_words(sentence_tokens_without_stopwords, vocabulary)
        if oov_words is None:
            continue
        print(f"Out-of-vocabulary words: {oov_words}")
        for wrong_word in oov_words:
            print(f"\nProcessing word: '{wrong_word}'")
            methods = ["jaccard", "frequency-based", "levenshtein", "jaro-winkler"]
            for method in methods:
                distances = calculate_distance(wrong_word, vocabulary, method, russian_alphabet)
                correction = find_correct_word(wrong_word, vocabulary, method, russian_alphabet)
                if distances:
                    top_candidates = sorted(distances.items(), key=lambda x: x[1])[:3]
                    print(f"{method}: '{wrong_word}' -> '{correction}'")
                    print(f"Top 3 candidates: {top_candidates}")
                    if method == "levenshtein" and correction:
                        print(f"Levenshtein matrix for '{wrong_word}' and '{correction}':")
                        filled_matrix = fill_levenshtein_matrix(wrong_word, correction)
                        if filled_matrix:
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
