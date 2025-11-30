"""
Auto-completion start
"""

# pylint:disable=unused-variable
from lab_3_generate_by_ngrams.main import BeamSearchTextGenerator, NGramLanguageModel, TextProcessor
from lab_4_auto_completion.main import WordProcessor, PrefixTrie


def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/hp_letters.txt", "r", encoding="utf-8") as letters_file:
        hp_letters = letters_file.read()
    with open("./assets/ussr_letters.txt", "r", encoding="utf-8") as text_file:
        ussr_letters = text_file.read()

    word_processor = WordProcessor('<EoW>')

    tokenized_text = word_processor._tokenize(hp_letters)
    encoded_sentences = word_processor.encode_sentences(tokenized_text)
    prefix_trie = PrefixTrie()
    prefix_trie.fill(encoded_sentences)
    suggestion = prefix_trie.suggest((2,))

    if suggestion:
        first_suggestion = suggestion[0]
        decoded_sequence = word_processor._postprocess_decoded_text(first_suggestion)
        print('First sequence:', decoded_sequence)
        result = decoded_sequence
    else:
        print('No continuations were found for the prefix')
    result = None
    assert result, "Result is None"


if __name__ == "__main__":
    main()
