"""
Auto-completion start
"""

from lab_3_generate_by_ngrams.main import BeamSearcher, BeamSearchTextGenerator, NGramLanguageModel
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
    result = my_trie.suggest((2,))
    print(word_processor.decode(result))
    assert result, "Result is None"


if __name__ == "__main__":
    main()
