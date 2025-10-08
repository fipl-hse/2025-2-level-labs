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
    find_out_of_vocab_words
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
    alphabet = ["а","б","в","г","д","е","ё","ж","з","и","й","к","л","м","н","о",
           "п","р","с","т","у","ф","х","ц","ч","ш","щ","ъ","ы","ь","э","ю","я",
           "А","Б","В","Г","Д","Е","Ё","Ж","З","И","Й","К","Л","М","Н","О",
           "П","Р","С","Т","У","Ф","Х","Ц","Ч","Ш","Щ","Ъ","Ы","Ь","Э","Ю","Я"]
    tokens=clean_and_tokenize(text)
    no_stop_words=remove_stop_words(tokens,stop_words)
    voc=build_vocabulary(no_stop_words) #valuable words from m&m and their frequencies
    #print(voc)
    tokens_in_sentences=[]
    for sentence in sentences:
        sentence=sentence.strip()
        for word in sentence.split():
            tokens_in_sentences.append(word)
    aliens=find_out_of_vocab_words(tokens_in_sentences,voc) #words that in sentences but not in m&m
    #print(aliens)
    corrections_jaccard={}
    corrections_freq_based={}
    for word in aliens:
        corrections_jaccard[word]=find_correct_word(word,voc,"jaccard",alphabet)
        corrections_freq_based[word]=find_correct_word(word,voc,"frequency-based",alphabet)
    print(corrections_jaccard)
    print(corrections_freq_based)
#    result = None
#    assert result, "Result is None"


if __name__ == "__main__":
    main()
