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
    result = None

    alphabet = list(map(chr, range(97, 123)))

    tokens = clean_and_tokenize(text) or []
    tokens_without_stopwords = remove_stop_words(tokens, stop_words) or []
    tokens_dict = build_vocabulary(tokens_without_stopwords) or {}
    out_of_vocab_words = find_out_of_vocab_words(tokens, tokens_dict) or []

    jaccard_distance = []
    print("calculating jaccard distances...")
    for word in out_of_vocab_words:
        jaccard_distance.append(find_correct_word(word, tokens_dict, "jaccard"))

    frequency_based_correct_word = []
    print("calculating frequency-based distances...")
    for word in out_of_vocab_words:
        frequency_based_correct_word.append(find_correct_word(word, tokens_dict,
                                                            "frequency-based", alphabet))

    levenshtein_correct_word = []
    print("calculating levenshtein distances...")
    for word in out_of_vocab_words:
        levenshtein_correct_word.append(find_correct_word(word, tokens_dict, "levenshtein"))
    
    jaro_winkler_correct_word = []
    print("calculating jaro-winkler distances...")
    for word in out_of_vocab_words:
        jaro_winkler_correct_word.append(find_correct_word(word, tokens_dict, "jaro-winkler"))

    for i in range(len(out_of_vocab_words)):
        print(f'''correct word for "{out_of_vocab_words[i]}":
            jaccard: {jaccard_distance[i]},
            frequency_based_correct_word: {frequency_based_correct_word[i]},
            levenshtein: {levenshtein_correct_word[i]},
            jaro_winkler_correct_word: {jaro_winkler_correct_word[i]}''')

    result = jaro_winkler_correct_word
    assert result, "Result is None"


if __name__ == "__main__":
    main()
