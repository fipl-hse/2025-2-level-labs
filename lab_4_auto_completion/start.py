"""
Auto-completion start
"""

from lab_4_auto_completion.main import PrefixTrie, WordProcessor

# pylint:disable=unused-variable


def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/hp_letters.txt", "r", encoding="utf-8") as letters_file:
        hp_letters = letters_file.read()
    with open("./assets/ussr_letters.txt", "r", encoding="utf-8") as text_file:
        ussr_letters = text_file.read()
    word_processor = WordProcessor(".")
    encoded_corpus = word_processor.encode_sentences(hp_letters)
    my_trie = PrefixTrie()
    my_trie.fill(encoded_corpus)
    suggestions = my_trie.suggest((2,))
    decoded_sequences = []
    for sequence in suggestions:
        decoded_sequences.append(word_processor.decode(sequence))
    result = tuple(decoded_sequences)
    print(result)
    assert result, "Result is None"


if __name__ == "__main__":
    main()
