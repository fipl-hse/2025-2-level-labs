"""
Spellcheck starter
"""

# pylint:disable=unused-variable, duplicate-code, too-many-locals
from lab_1_keywords_tfidf.main import clean_and_tokenize, remove_stop_words
from lab_2_spellcheck.main import build_vocabulary, find_correct_word, find_out_of_vocab_words, calculate_jaccard_distance

russian = list("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")

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

#----------------------------------------------------
    tokenized_corpus = clean_and_tokenize(text)
    if not tokenized_corpus:
        return

    corpus_without_stopwords = remove_stop_words(tokenized_corpus, stop_words)
    if not corpus_without_stopwords:
        return

    tokenized_sentences = clean_and_tokenize(''.join(sentences))
    if not tokenized_sentences:
        return
    
    sentences_without_stopwords = remove_stop_words(tokenized_sentences, stop_words)
    if not sentences_without_stopwords:
        return

    vocabulary = build_vocabulary(corpus_without_stopwords)
    if not vocabulary:
        return
    #print("Vocabulary: ", vocabulary, "\n")

    sentence_to_get_10 = "В белом плаще с кровавым подбоем, шаркающей кавалерийской походкой," \
    " ранним утром четырнадцатого числа весеннего месяца нисана в крытую колоннаду" \
    " между двумя крыльями дворца Ирода Великого вышел прокуратор Иудеи Понтий Пилат."

    tokenized_sentence = clean_and_tokenize(sentence_to_get_10)
    if not tokenized_sentence:
        return

    sentence_without_stopwords = remove_stop_words(tokenized_sentence, stop_words)
    print("Tokens: ", sentence_without_stopwords)

    words_out_of_voc = find_out_of_vocab_words(sentence_without_stopwords, vocabulary)
    if not words_out_of_voc:
        return

    correct_words = []
    total_distance = 0.0

    for word in sentence_without_stopwords:
        correct_word = find_correct_word(word, vocabulary, "jaccard", russian)

        if correct_word:
            distance = calculate_jaccard_distance(word, correct_word)
            if distance is not None:
                correct_words.append(correct_word)
                total_distance += distance

    print(correct_words, total_distance)

    #tokens_out_of_voc = find_out_of_vocab_words(sentences_without_stopwords, vocabulary)
    #if not tokens_out_of_voc:
    #    return
    #print("Tokens out of vocabulary: ", tokens_out_of_voc, "\n")

    #correct_words_by_jacc = {token: find_correct_word(
    #    token, vocabulary, "jaccard", russian)
    #    for token in tokens_out_of_voc}
    #print("Correct words by jaccard method: ", correct_words_by_jacc, "\n")

    #correct_words_by_freq = {token: find_correct_word(
    #    token, vocabulary, "frequency-based", russian)
    #    for token in tokens_out_of_voc}
    #print("Correct words by frequensy based method: ", correct_words_by_freq, "\n")

    #correct_words_by_lev = {token: find_correct_word(
    #    token, vocabulary, "levenshtein", russian)
    #    for token in tokens_out_of_voc}
    #print("Correct words by levenshtein method: ", correct_words_by_lev, "\n")

    #correct_words_by_jaro = {token: find_correct_word(
    #    token, vocabulary, "jaro-winkler", russian)
    #    for token in tokens_out_of_voc}
    #print("Correct words by jaro-winkler method: ", correct_words_by_jaro, "\n")

    #result = correct_words_by_jaro
    #assert result, "Result is None"


if __name__ == "__main__":
    main()
