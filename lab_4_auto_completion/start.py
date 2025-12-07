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
    encoded_texts = {
        'hp': processor.encode_sentences(hp_letters),
        'ussr': processor.encode_sentences(ussr_letters)
    }
    trie = PrefixTrie()
    trie.fill(encoded_texts['hp'])
    suggestion = trie.suggest((2,))[0]
    print(f"\nDecoded output: {processor.decode(suggestion)}")
    model = NGramTrieLanguageModel(encoded_texts['hp'], 5)
    model.build()
    generation_results = {}
    generation_results['before'] = {
        'greedy': GreedyTextGenerator(model, processor).run(52, 'Dear'),
        'beam': BeamSearchTextGenerator(model, processor, 3).run('Dear', 52)
    }
    model.update(encoded_texts['ussr'])
    generation_results['after'] = {
        'greedy': GreedyTextGenerator(model, processor).run(52, 'Dear'),
        'beam': BeamSearchTextGenerator(model, processor, 3).run('Dear', 52)
    }
    print(f"\nGreedy before: {generation_results['before']['greedy']}")
    print(f"Beam before: {generation_results['before']['beam']}")
    print(f"\nGreedy after: {generation_results['after']['greedy']}")
    print(f"Beam after: {generation_results['after']['beam']}")
    dynamic = DynamicNgramLMTrie(encoded_texts['hp'], 5)
    dynamic.build()
    save(dynamic, "./saved_dynamic_trie.json")
    loaded = load("./saved_dynamic_trie.json")
    generator = DynamicBackOffGenerator(loaded, processor)
    print(f"\nDynamic before: {generator.run(50, 'Ivanov')}")
    loaded.update(encoded_texts['ussr'])
    size = 3
    try:
        loaded.set_current_ngram_size(size)
    except IncorrectNgramError:
        loaded.set_current_ngram_size(None)
    dynamic_result = generator.run(50, 'Ivanov')
    print(f"Dynamic after: {dynamic_result}\n")
    assert dynamic_result, "Result is None"

if __name__ == "__main__":
    main()
