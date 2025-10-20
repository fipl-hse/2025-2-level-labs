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

    alphabet = list("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")

    tokens = clean_and_tokenize(text) or []
    tokens_without_stopwords = remove_stop_words(tokens, stop_words) or []
    vocabulary = build_vocabulary(tokens_without_stopwords) or {}

    correct_words = ["спорили","шумном", "патритича", "прохожие", "усталый",
                    "саду","моста", "ждала", "обсуждая", "сидели",
                    "принёс","толстую","тёмной","записывал",
                    "представления", "найти","литературы","одобрение",
                    "двое","вечернего","заскрипел","надеясь","деревянной"]

    tokenized_sentences = list(set(token
    for sentence in sentences
    for token in remove_stop_words(clean_and_tokenize(sentence) or [], stop_words)
      ))

    out_of_vocab_words = find_out_of_vocab_words(tokenized_sentences, vocabulary) or []

    for word in correct_words:
        if word in out_of_vocab_words:
            out_of_vocab_words.remove(word)

    print(" These mispelled words are out of vocabulary:\n", out_of_vocab_words)


    final_corrections = {}

    for word in out_of_vocab_words:
        print(f'\nCorrections for the word "{word}"')
        corrections = {}
        jaccard_correction = find_correct_word(word, vocabulary, "jaccard", alphabet)
        corrections['jaccard'] = jaccard_correction
        print(f' By Jaccard: "{jaccard_correction}"')

        frequency_based_correction = find_correct_word(word, vocabulary,"frequency-based", alphabet)
        corrections['frequency-based'] = frequency_based_correction
        print(f' By frequency: "{frequency_based_correction}"')

        levenshtein_correction = find_correct_word(word, vocabulary, "levenshtein", alphabet)
        corrections['levenshtein'] = levenshtein_correction
        print(f' By Levenshtein: "{levenshtein_correction}"')
        final_corrections[word] = corrections

    result = final_corrections
    print(f'\n Total results: {result}')

    assert result, "Result is None"


if __name__ == "__main__":
    main()
