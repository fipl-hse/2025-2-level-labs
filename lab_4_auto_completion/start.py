"""
Auto-completion start
"""

# pylint:disable=unused-variable
from lab_3_generate_by_ngrams.main import BeamSearcher, BeamSearchTextGenerator, GreedyTextGenerator
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
    with open("./assets/secrets/secret_5.txt", "r", encoding="utf-8") as text_file:
        letter = text_file.read()
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as harry_file:
        text = harry_file.read()
    processor = WordProcessor('<EoS>')
    hp_encoded = processor.encode_sentences(hp_letters)

    trie = PrefixTrie()
    trie.fill(hp_encoded)
    suggestions = trie.suggest((2,))
    if suggestions:
        first = suggestions[0]
        words = []
        for code in first:
            for text, el in processor._storage.items():
                if el == code:
                    words.append(text)
                    break
    decoded_text = processor._postprocess_decoded_text(tuple(words))
    print(f"\n1. Decoded result: {decoded_text}")

    model = NGramTrieLanguageModel(hp_encoded, 5)
    model.build()

    greedy_result = GreedyTextGenerator(model, processor).run(52, 'Harry Potter')
    print(f"\n2. Greedy result before: {greedy_result}")
    beam_result = BeamSearchTextGenerator(model, processor, 3).run('Harry Potter', 52)
    print(f"Beam result before: {beam_result}")

    encoded_ussr = processor.encode_sentences(ussr_letters)
    model.update(encoded_ussr)

    greedy_updated = GreedyTextGenerator(model, processor).run(52, 'Harry Potter')
    print(f"\n3. Greedy result after: {greedy_updated}")
    beam_updated = BeamSearchTextGenerator(model, processor, 3).run('Harry Potter', 52)
    print(f"Beam result before: {beam_updated}")

    dynamic_trie = DynamicNgramLMTrie(hp_encoded, 5)
    dynamic_trie.build()

    save_path = "./saved_dynamic_trie.json"
    save(dynamic_trie, save_path)
    loaded_trie = load(save_path)

    dynamic_generator = DynamicBackOffGenerator(loaded_trie, processor)
    dynamic_result_before = dynamic_generator.run(50, 'Ivanov')
    print(f"\n4. Dynamic result before: {dynamic_result_before}")

    loaded_trie.update(encoded_ussr)
    loaded_trie.set_current_ngram_size(3)
    try:
        loaded_trie.set_current_ngram_size(3)
    except IncorrectNgramError:
        loaded_trie.set_current_ngram_size(None)

    dynamic_result_after = dynamic_generator.run(50, 'Ivanov')
    print(f"Dynamic result after: {dynamic_result_after}\n")

    result = dynamic_result_after
    assert result, "Result is None"

if __name__ == "__main__":
    main()
