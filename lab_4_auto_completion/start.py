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
    print(f"\nDecoded output: {processor.decode(suggestion)}")
    model = NGramTrieLanguageModel(encoded_hp, 5)
    model.build()
    greedy_before = GreedyTextGenerator(model, processor).run(52, 'Dear')
    beam_before = BeamSearchTextGenerator(model, processor, 3).run('Dear', 52)
    print(f"\nGreedy before: {greedy_before}")
    print(f"Beam before: {beam_before}")
    encoded_ussr = processor.encode_sentences(ussr_letters)
    model.update(encoded_ussr)
    greedy_after = GreedyTextGenerator(model, processor).run(52, 'Dear')
    beam_after = BeamSearchTextGenerator(model, processor, 3).run('Dear', 52)
    print(f"\nGreedy after: {greedy_after}")
    print(f"Beam after: {beam_after}")
    dynamic = DynamicNgramLMTrie(encoded_hp, 5)
    dynamic.build()
    save(dynamic, "./saved_dynamic_trie.json")
    loaded = load("./saved_dynamic_trie.json")
    generator = DynamicBackOffGenerator(loaded, processor)
    print(f"\nDynamic before: {generator.run(50, 'Ivanov')}")
    loaded.update(encoded_ussr)
    size = 3
    max_size = loaded._max_ngram_size
    if 2 <= size <= max_size:
        loaded.set_current_ngram_size(size)
    else:
        loaded.set_current_ngram_size(max_size)
    dynamic_result = generator.run(50, 'Ivanov')
    print(f"Dynamic after: {dynamic_result}\n")
    assert dynamic_result, "Result is None"

if __name__ == "__main__":
    main()
