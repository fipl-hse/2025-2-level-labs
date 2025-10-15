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
    no_stop_words = remove_stop_words(tokens,stop_words) or []
    voc = build_vocabulary(no_stop_words) or {}
    #print(voc)
    tokens_in_sentences = []
    for sentence in sentences:
        sentence_tokens = clean_and_tokenize(sentence) or []
        sentence_no_stop_words = remove_stop_words(sentence_tokens,stop_words) or []
        tokens_in_sentences.extend(sentence_no_stop_words)
    aliens = find_out_of_vocab_words(tokens_in_sentences,voc) or []
    print(aliens)
    alphabet = ["а","б","в","г","д","е","ё","ж","з","и","й","к","л","м","н","о",
           "п","р","с","т","у","ф","х","ц","ч","ш","щ","ъ","ы","ь","э","ю","я"]
    sum_results = {}
    for token in aliens:
        corrections_jaccard = find_correct_word(token,voc,"jaccard",alphabet) or {}
        corrections_freq_based = find_correct_word(token,voc,"frequency-based",alphabet) or {}
        corrections_levenshtein = find_correct_word(token,voc,"levenshtein",alphabet) or {}
        sum_results[token] = {
            'jaccard': corrections_jaccard,
            'frequency-based': corrections_freq_based,
            'levenshtein': corrections_levenshtein,
        }
    print(sum_results)
    result = sum_results
    assert result, "Result is None"


if __name__ == "__main__":
    main()
