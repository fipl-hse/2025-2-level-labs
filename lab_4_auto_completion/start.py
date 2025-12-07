"""
Auto-completion start
"""

# pylint:disable=unused-variable
from lab_3_generate_by_ngrams.main import (
    BeamSearchTextGenerator,
    GreedyTextGenerator,
)
from lab_4_auto_completion.main import (
    DynamicBackOffGenerator,
    DynamicNgramLMTrie,
    IncorrectNgramError,
    load,
    NGramTrieLanguageModel,
    PrefixTrie,
    save,
    WordProcessor,
)


def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/hp_letters.txt", "r", encoding="utf-8") as letters_file:
        hp_letters = letters_file.read()
    with open("./assets/ussr_letters.txt", "r", encoding="utf-8") as text_file:
        ussr_letters = text_file.read()
    result = None
    assert result, "Result is None"

    processor = WordProcessor('<EOS>')
    encoded_sentences = processor.encode_sentences(hp_letters)

    prefix_trie = PrefixTrie()
    prefix_trie.fill(encoded_sentences)
    suggestions = prefix_trie.suggest((2,))
    if suggestions:
        print(f"Decoded: {processor.decode(suggestions[0]).replace("<EOS>", "").strip()}")

    model = NGramTrieLanguageModel(encoded_sentences, 5)
    model.build()

    print(f"Previous result of Greedy: {GreedyTextGenerator(model, processor).run(52, 'Harry')}")

    print(f"Previous result of Beam: {BeamSearchTextGenerator(model, processor, 3).run('Harry', 52)}")

    encoded_ussr_sentences = processor.encode_sentences(ussr_letters)
    model.update(encoded_ussr_sentences)

    print(f"Actual result of Greedy: {GreedyTextGenerator(model, processor).run(52, 'Harry')}")

    print(f"Actual result of Beem: {BeamSearchTextGenerator(model, processor, 3).run('Harry', 52)}")

    dynamic_trie = DynamicNgramLMTrie(encoded_sentences, 5)
    dynamic_trie.build()

    save(dynamic_trie, "./saved_dynamic_trie.json")
    loaded_trie = load("./saved_dynamic_trie.json")

    dynamic_generator = DynamicBackOffGenerator(loaded_trie, processor)
    print(f"Dynamic before: {dynamic_generator.run(50, 'Ivanov')}")

    loaded_trie.update(encoded_ussr_sentences)
    loaded_trie.set_current_ngram_size(3)
    try:
        loaded_trie.set_current_ngram_size(3)
    except IncorrectNgramError:
        loaded_trie.set_current_ngram_size(None)

    print(f"Actual Dynamik: {dynamic_generator.run(50, 'Ivanov')}")

    result = dynamic_generator
    assert result, "Result is None"

if __name__ == "__main__":
    main()
