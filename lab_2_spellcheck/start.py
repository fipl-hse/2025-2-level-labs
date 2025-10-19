"""
Spellcheck starter
"""

# pylint:disable=unused-variable, duplicate-code, too-many-locals
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
    tokens = clean_and_tokenize(text) or []
    tokens_without_stopwords = remove_stop_words(tokens, stop_words) or []
    vocabulary = build_vocabulary(tokens_without_stopwords) or {}
    all_sentence_tokens = []
    for sentence in sentences:
        sentence_tokens = clean_and_tokenize(sentence) or []
        sentence_tokens_without_stop_words = remove_stop_words(sentence_tokens, stop_words) or []
        all_sentence_tokens.extend(sentence_tokens_without_stop_words)
    wrong_words = find_out_of_vocab_words(all_sentence_tokens, vocabulary) or []
    print(wrong_words)
    alphabet = [chr(i) for i in range(1072, 1104)]
    all_results = {}
    for wrong_word in wrong_words:
        print(f"\ncorrection for '{wrong_word}':")
        jaccard_correction = find_correct_word(wrong_word, vocabulary,
                                               'jaccard', alphabet) or {}
        print(f"  Jaccard: {jaccard_correction}")
        all_results[wrong_word] = {'jaccard': jaccard_correction}
    result = all_results
    assert result, "Result is None"


if __name__ == "__main__":
    main()
