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
        senten_words = ''.join(sentences)
    cleaned_text = clean_and_tokenize(text)
    if cleaned_text is None:
        return
    removed_stop_words = remove_stop_words(cleaned_text, stop_words)
    if removed_stop_words is None:
        return
    cleaned_sentences = clean_and_tokenize(senten_words)
    if cleaned_sentences is None:
        return
    removed_sentences_stop_words = (
        remove_stop_words(cleaned_sentences, stop_words)
        )
    if removed_sentences_stop_words is None:
        return
    freq_vocab = build_vocabulary(removed_stop_words)
    if freq_vocab is None:
        return
    incorrect_words = (
        find_out_of_vocab_words(removed_sentences_stop_words, freq_vocab)
        )


    print(incorrect_words)
    alphabet = list("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")
    jaccard=[]
    frequency_based=[]
    levenshtein=[]
    for incorrect_word in incorrect_words:
        jaccard_correct_words = find_correct_word(incorrect_word, freq_vocab, "jaccard")
        jaccard.append(jaccard_correct_words)
        frequency_based_correct_words = (
            find_correct_word(incorrect_word, freq_vocab, "frequency-based", alphabet)
            )
        frequency_based.append(frequency_based_correct_words)
        lev_correct_words = find_correct_word(incorrect_word, freq_vocab, "levenshtein")
        levenshtein.append(lev_correct_words)
    correct_words = (
        f"JACCARD : {jaccard} \nFREQUENCY_BAESD: {frequency_based} \nLEVENSHTEIN: {levenshtein}"
        )
    print(correct_words)
    result = correct_words
    assert result, "Result is None"

if __name__ == "__main__":
    main()
