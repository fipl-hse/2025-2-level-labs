"""
Auto-completion start
"""

# pylint:disable=unused-variable

from lab_3_generate_by_ngrams.main import BeamSearchTextGenerator, GreedyTextGenerator
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

    processor = WordProcessor('<EoS>')
    hp_encoded = processor.encode_sentences(hp_letters)

    trie = PrefixTrie()
    trie.fill(hp_encoded)
    suggestion = trie.suggest((2,))[0]
    print(f" \n1.Decoded result: {processor.decode(suggestion)}")

    lm = NGramTrieLanguageModel(hp_encoded, 5)
    lm.build()

    print(f"\n 2.Greedy before merging: {GreedyTextGenerator(lm, processor).run(52, 'Dear')}")
    print(f"Beam before merging: {BeamSearchTextGenerator(lm, processor, 3).run('Dear', 52)}")

    print("\nMerging corpuses...")
    encoded_ussr = processor.encode_sentences(ussr_letters)
    lm.update(encoded_ussr)

    print(f"\n3.Greedy after merging: {GreedyTextGenerator(lm, processor).run(52, 'Dear')}")
    beam_updated = BeamSearchTextGenerator(lm, processor, 3).run('Dear', 52)
    print(f"Beam after merging: {beam_updated}")


    dynamic_trie = DynamicNgramLMTrie(hp_encoded, 5)
    dynamic_trie.build()

    save(dynamic_trie, r"./assets/dynamic_trie.json")
    loaded_trie = load(r"./assets/dynamic_trie.json")

    dynamic_generator = DynamicBackOffGenerator(loaded_trie, processor)
    print(f"\n4. Dynamic back off before merging: {dynamic_generator.run(50, 'Ivanov')}")

    loaded_trie.update(encoded_ussr)
    loaded_trie.set_current_ngram_size(3)
    try:
        loaded_trie.set_current_ngram_size(3)
    except IncorrectNgramError:
        loaded_trie.set_current_ngram_size(None)

    print(f"Dynamic back off after merging: {dynamic_generator.run(50, 'Ivanov')}\n")

    result = dynamic_generator
    assert result, "Result is None"

if __name__ == "__main__":
    main()
