"""
Auto-completion start
"""

# pylint:disable=unused-variable
from lab_3_generate_by_ngrams.main import BeamSearchTextGenerator, GreedyTextGenerator
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
    decoded_suggestion = processor.decode(suggestions[0])
    result = decoded_suggestion

    model = NGramTrieLanguageModel(encoded_sentences, 5)
    model.build()

    ussr_encoded = processor.encode_sentences(ussr_letters)
    model.update(ussr_encoded)

    greedy_generator = GreedyTextGenerator(model, processor)
    beam_generator = BeamSearchTextGenerator(model, processor, 3)

    result_generator = greedy_generator.run(51, "Harry ")
    print(result_generator)

    print(result)
    assert result, "Result is None"


if __name__ == "__main__":
    main()
