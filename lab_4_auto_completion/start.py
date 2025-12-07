"""
Auto-completion start
"""
from lab_4_auto_completion.main import (WordProcessor, PrefixTrie)

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
    encoding_letters = word_processor.encode_sentences(hp_letters)
    prefix_trie = PrefixTrie()
    prefix_trie.fill(encoding_letters)
    suggestions = prefix_trie.suggest((2,))
    if suggestions:
        main_suggestion = suggestions[0]
        print(word_processor.decode(main_suggestion))
    else:
        print("No suggestions")
    result = main_suggestion
    assert result, "Result is None"


if __name__ == "__main__":
    main()
