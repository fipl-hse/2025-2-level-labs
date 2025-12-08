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
    processor = WordProcessor('<EoS>')
    encoded_hp = processor.encode_sentences(hp_letters)
    trie = PrefixTrie()
    trie.fill(encoded_hp)
    suggestion = trie.suggest((2,))[0]
    print(f"\n1. Decoded result: {processor.decode(suggestion)}")
    model = NGramTrieLanguageModel(encoded_hp, 5)
    model.build()
    print(f"\n2. Greedy result before: {GreedyTextGenerator(model, processor).run(52, 'Dear')}")
    print(f"Beam result before: {BeamSearchTextGenerator(model, processor, 3).run('Dear', 52)}")
    encoded_ussr = processor.encode_sentences(ussr_letters)
    model.update(encoded_ussr)
    print(f"\n3. Greedy result after: {GreedyTextGenerator(model, processor).run(52, 'Dear')}")
    beam_updated = BeamSearchTextGenerator(model, processor, 3).run('Dear', 52)
    print(f"Beam result after: {beam_updated}")
    dynamic = DynamicNgramLMTrie(encoded_hp, 5)
    dynamic.build()
    save(dynamic, "./saved_dynamic_trie.json")
    loaded = load("./saved_dynamic_trie.json")
    generator = DynamicBackOffGenerator(loaded, processor)
    print(f"\n4. Dynamic result before: {generator.run(50, 'Ivanov')}")
    loaded.update(encoded_ussr)
    loaded.set_current_ngram_size(3)
    try:
        loaded.set_current_ngram_size(3)
    except IncorrectNgramError:
        loaded.set_current_ngram_size(None)
    print(f"Dynamic result after: {generator.run(50, 'Ivanov')}\n")
    result = generator
    assert result, "Result is None"

if __name__ == "__main__":
    main()
