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
    result = ""
    processor = WordProcessor('<EOS>')
    encoded_sentences = processor.encode_sentences(hp_letters)
    prefix_trie = PrefixTrie()
    prefix_trie.fill(encoded_sentences)
    suggestions = prefix_trie.suggest((2,))
    if suggestions:
        decoded = processor.decode(suggestions[0])
        result = decoded.replace("<EOS>", "").strip()
        print(result) 
    assert result is not None and result != "", "Result is None or empty"


if __name__ == "__main__":
    main()
