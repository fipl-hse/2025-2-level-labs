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
    with open("./assets/hp_letters.txt","r",encoding="utf-8") as letters_file:
        hp_letters = letters_file.read()
    with open("./assets/ussr_letters.txt", "r", encoding="utf-8") as text_file:
        ussr_letters = text_file.read()
    processor = WordProcessor('<EoS>')
    hp_encoded = processor.encode_sentences(hp_letters)

    trie = PrefixTrie()
    trie.fill(hp_encoded)
    suggestion = trie.suggest((2,))[0]
    print(f"\n1. Decoded result: {processor.decode(suggestion)}")

    model = NGramTrieLanguageModel(hp_encoded, 5)
    model.build()

    print(
        f"\n2. Greedy result before: "
        f"{GreedyTextGenerator(model, processor).run(52, 'Dear')}"
    )
    print(
        f"Beam result before: "
        f"{BeamSearchTextGenerator(model, processor, 3).run('Dear', 52)}"
    )

    encoded_ussr = processor.encode_sentences(ussr_letters)
    model.update(encoded_ussr)

    print(
        f"\n3. Greedy result after: "
        f"{GreedyTextGenerator(model, processor).run(52, 'Dear')}"
    )
    beam_updated = BeamSearchTextGenerator(model, processor, 3).run('Dear', 52)
    print(f"Beam result before: {beam_updated}")

    dynamic_trie = DynamicNgramLMTrie(hp_encoded, 5)
    dynamic_trie.build()

    save(dynamic_trie, "./saved_dynamic_trie.json")
    loaded_trie = load("./saved_dynamic_trie.json")

    dynamic_generator = DynamicBackOffGenerator(loaded_trie, processor)
    print(f"\n4. Dynamic result before: {dynamic_generator.run(50, 'Ivanov')}")

    loaded_trie.update(encoded_ussr)
    loaded_trie.set_current_ngram_size(3)
    try:
        loaded_trie.set_current_ngram_size(3)
    except IncorrectNgramError:
        loaded_trie.set_current_ngram_size(None)

    print(f"Dynamic result after: {dynamic_generator.run(50, 'Ivanov')}\n")

    result = dynamic_generator
    assert result, "Result is None"

    if __name__ == "__main__":
        main()
