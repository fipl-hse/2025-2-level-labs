"""
Auto-completion start
"""

# pylint:disable=unused-variable
from lab_4_auto_completion.main import NGramTrieLanguageModel, PrefixTrie, WordProcessor


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
    print(suggestions[0])
    decoded_suggestion = processor.decode(suggestions[0])
    result = decoded_suggestion
    model = NGramTrieLanguageModel(encoded_sentences, 5)
    model.build()
    print(result)
    assert result, "Result is None"


if __name__ == "__main__":
    main()
