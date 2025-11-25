"""
Auto-completion start
"""

# pylint:disable=unused-variable
from lab_4_auto_completion.main import PrefixTrie, WordProcessor


def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/hp_letters.txt", "r", encoding="utf-8") as letters_file:
        hp_letters = letters_file.read()
    with open("./assets/ussr_letters.txt", "r", encoding="utf-8") as text_file:
        ussr_letters = text_file.read()
    word_processor = WordProcessor(end_of_sentence_token="<EOS>")
    encoded_sentences = word_processor.encode_sentences(hp_letters)
    all_words = []
    for sentence in encoded_sentences:
        all_words.extend(sentence)
    n_gram_size = 3
    trigrams = []
    for i in range(len(all_words) - n_gram_size + 1):
        trigram = tuple(all_words[i:i + n_gram_size])
        trigrams.append(trigram)
    trie = PrefixTrie()
    trie.fill(tuple(trigrams))
    prefix = (2,)
    suggestions = trie.suggest(prefix)
    print(f"Found {len(suggestions)} suggestions for prefix {prefix}")
    if suggestions:
        first_suggestion = suggestions[0]
        print(f"First suggestion: {first_suggestion}")
        decoded_words = []
        for token in first_suggestion:
            for word, word_id in word_processor._storage.items():
                if word_id == token:
                    decoded_words.append(word)
                    break
        decoded_text = word_processor._postprocess_decoded_text(tuple(decoded_words))
        print(f"Decoded result: {decoded_text}")
    result = decoded_text
    assert result, "Result is None"


if __name__ == "__main__":
    main()
