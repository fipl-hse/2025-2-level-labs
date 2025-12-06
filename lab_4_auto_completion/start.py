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

    processor = WordProcessor('<EOS>')
    encoded_sentences = processor.encode_sentences(hp_letters)

    prefix_trie = PrefixTrie()
    prefix_trie.fill(encoded_sentences)
    suggestions = prefix_trie.suggest((2,))
    if suggestions:
        decoded = processor.decode(suggestions[0])
        print(decoded.replace("<EOS>", "").strip())

    model = NGramTrieLanguageModel(encoded_sentences, 5)
    model.build()

    greedy_before = GreedyTextGenerator(model, processor)
    gb_result = greedy_before.run(52, 'Harry Potter')
    print(f"Greedy Generator befor: {gb_result}")

    beam_before = BeamSearchTextGenerator(model, processor, 3)
    bb_result = beam_before.run('Harry Potter', 52)
    print(f"Beam Generator befor: {bb_result}")

    encoded_ussr_sentences = processor.encode_sentences(ussr_letters)
    model.update(encoded_ussr_sentences)

    greedy_after = GreedyTextGenerator(model, processor)
    ga_result = greedy_after.run(52, 'Harry Potter')
    print(f"Greedy Generator after: {ga_result}")

    beam_after = BeamSearchTextGenerator(model, processor, 3)
    ba_result = beam_after.run('Harry Potter', 52)
    print(f"Beam Generator after: {ba_result}")

    dynamic_trie = DynamicNgramLMTrie(encoded_sentences, 5)
    dynamic_trie.build()

    save_path = "./saved_dynamic_trie.json"
    save(dynamic_trie, save_path)
    loaded_trie = load(save_path)

    dynamic_generator = DynamicBackOffGenerator(loaded_trie, processor)
    dynamic_result_before = dynamic_generator.run(50, 'Ivanov')

    loaded_trie.update(encoded_ussr_sentences)
    loaded_trie.set_current_ngram_size(5)
    try:
        loaded_trie.set_current_ngram_size(5)
    except IncorrectNgramError:
        loaded_trie.set_current_ngram_size(None)

    dynamic_result_after = dynamic_generator.run(50, 'Ivanov')

    result = (dynamic_result_before, dynamic_result_after)
    assert result, "Result is None"


if __name__ == "__main__":
    main()
