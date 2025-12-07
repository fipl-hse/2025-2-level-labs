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
    hp_encoded = processor.encode_sentences(hp_letters)
    encoded_ussr = processor.encode_sentences(ussr_letters)
    trie = PrefixTrie()
    trie.fill(hp_encoded)
    suggestion = trie.suggest((2,))[0]
    print(f"\nDecoded output: {processor.decode(suggestion)}")
    model = NGramTrieLanguageModel(hp_encoded, 5)
    model.build()
    print(f"\nGreedy before: {GreedyTextGenerator(model, processor).run(52, 'Dear')}")
    beam_before = BeamSearchTextGenerator(model, processor, 3).run('Dear', 52)
    print(f"Beam before: {beam_before}")
    model.update(encoded_ussr)
    print(f"\nGreedy after: {GreedyTextGenerator(model, processor).run(52, 'Dear')}")
    beam_after = BeamSearchTextGenerator(model, processor, 3).run('Dear', 52)
    print(f"Beam after: {beam_after}")
    dynamic = DynamicNgramLMTrie(hp_encoded, 5)
    dynamic.build()
    save(dynamic, "./saved_dynamic_trie.json")
    loaded = load("./saved_dynamic_trie.json")
    generator = DynamicBackOffGenerator(loaded, processor)
    print(f"\nDynamic before: {generator.run(50, 'Ivanov')}")
    loaded.update(encoded_ussr)
    try:
        loaded.set_current_ngram_size(3)
    except IncorrectNgramError:
        loaded.set_current_ngram_size(None)
    dynamic_result = generator.run(50, 'Ivanov')
    print(f"Dynamic after: {dynamic_result}\n")
    assert dynamic_result, "Result is None"

if __name__ == "__main__":
    main()
