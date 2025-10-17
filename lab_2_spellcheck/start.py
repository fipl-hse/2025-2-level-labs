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
    tokens = clean_and_tokenize(text)
    if tokens is None:
        return
    tokens_without_stopwords = remove_stop_words(tokens, stop_words)
    if tokens_without_stopwords is None:
        return
    vocab = build_vocabulary(tokens_without_stopwords)
    if vocab is None:
        return
    corrections = []
    for sentence in sentences:
        words = clean_and_tokenize(sentence)
        if words is None:
            continue
        out_of_vocab = find_out_of_vocab_words(words, vocab)
        if out_of_vocab is None:
            continue
        for word in out_of_vocab:
            use_jaccard = find_correct_word(word, vocab, "jaccard")
            use_frequency_based = find_correct_word(word, vocab, "frequency-based")
            use_levenshtein = find_correct_word(word, vocab, "levenshtein")
            corrections.append({
                "word": word, 
                "jaccard": use_jaccard, 
                "frequency-based": use_frequency_based,
                "levenshtein": use_levenshtein
            })
            print(f"Word: {word}")
            print(f"Jaccard: {use_jaccard}")
            print(f"Frequency-based: {use_frequency_based}")
            print(f"Levenshtein: {use_levenshtein}")
    result = corrections
    assert result, "Result is None"


if __name__ == "__main__":
    main()
