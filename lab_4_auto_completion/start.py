"""
Auto-completion start
"""

# pylint:disable=unused-variable
from lab_4_auto_completion.main import DecodingError, PrefixTrie, WordProcessor


def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/hp_letters.txt", "r", encoding="utf-8") as letters_file:
        hp_letters = letters_file.read()
    with open("./assets/ussr_letters.txt", "r", encoding="utf-8") as text_file:
        ussr_letters = text_file.read()
    processor = WordProcessor('<EOS>')
    encoded_sentences = processor.encode_sentences(hp_letters)

    prefix_trie = PrefixTrie()
    prefix_trie.fill(encoded_sentences)
    suggestions = prefix_trie.suggest((2,))
    if suggestions:
        first_suggestion = suggestions[0]
        decoded_words = []
        for word_ind in first_suggestion:
            word = None
            for w, w_ind in processor._storage.items():
                if w_ind == word_ind:
                    word = w
                    break
            if word is not None:
                decoded_words.append(word)
            else:
                decoded_words.append(f"[ID:{word_ind}]")

        decoded_sequence = processor._postprocess_decoded_text(tuple(decoded_words))
        print(f"First sequence: {decoded_sequence}")
        result = decoded_sequence
    else:
        print("No suggestions were found")
        result = None

    #result = None
    #assert result, "Result is None"


if __name__ == "__main__":
    main()
