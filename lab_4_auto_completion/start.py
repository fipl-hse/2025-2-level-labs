"""
Auto-completion start
"""

# pylint:disable=unused-variable
from lab_3_generate_by_ngrams.main import BeamSearchTextGenerator, GreedyTextGenerator
from lab_4_auto_completion.main import (
    DynamicBackOffGenerator,
    DynamicNgramLMTrie,
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
    processor = WordProcessor('<EoS>')
    encoded_hp = processor.encode_sentences(hp_letters)
    trie = PrefixTrie()
    trie.fill(encoded_hp)
    suggestion = trie.suggest((2,))[0]
    print(f"Decoded: {processor.decode(suggestion)}")
    model = NGramTrieLanguageModel(encoded_hp, 5)
    model.build()
    print(f"Greedy first: {GreedyTextGenerator(model, processor).run(52, 'Dear')}")
    print(f"Beam first: {BeamSearchTextGenerator(model, processor, 3).run('Dear', 52)}")
    encoded_ussr = processor.encode_sentences(ussr_letters)
    model.update(encoded_ussr)
    print(f"Greedy second: {GreedyTextGenerator(model, processor).run(52, 'Dear')}")
    beam_updated = BeamSearchTextGenerator(model, processor, 3).run('Dear', 52)
    print(f"Beam second: {beam_updated}")
    dynamic = DynamicNgramLMTrie(encoded_hp, 5)
    dynamic.build()
    save(dynamic, "./saved_dynamic_trie.json")
    loaded = load("./saved_dynamic_trie.json")
    generator = DynamicBackOffGenerator(loaded, processor)
    print(f"Dynamic first: {generator.run(50, 'Ivanov')}")
    loaded.update(encoded_ussr)
    loaded.set_current_ngram_size(3)
    loaded.set_current_ngram_size(3)
    print(f"Dynamic second: {generator.run(50, 'Ivanov')}")
    result = generator
    assert result, "Result is None"


if __name__ == "__main__":
    main()
