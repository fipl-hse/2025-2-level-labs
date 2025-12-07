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

    word_processor = WordProcessor('<EoW>')
    encoded_hp = word_processor.encode_sentences(hp_letters)
    tree = PrefixTrie()
    tree.fill(encoded_hp)

    if (found := tree.suggest((2,))):
        print(f"Found {len(found)} suggestions, first: {word_processor.decode(found[0])}")
    else:
        print('No continuations found')

    model = NGramTrieLanguageModel(encoded_hp, 5)
    model.build()

    print(f'Greedy: {GreedyTextGenerator(model, word_processor).run(52, "Dear")}')
    print(f'Beam: {BeamSearchTextGenerator(model, word_processor, 3).run("Dear", 52)}')

    encoded = word_processor.encode_sentences(ussr_letters)
    model.update(encoded)

    print(f'Greedy update: {GreedyTextGenerator(model, word_processor).run(52, "Dear")}')
    print(f'Beam update: {BeamSearchTextGenerator(model, word_processor, 3).run("Dear", 52)}')

    dynamic_model = DynamicNgramLMTrie(encoded_hp, 5)
    dynamic_model.build()

    save(dynamic_model, "./dynamic_model.json")
    loaded_file = load("./dynamic_model.json")

    loaded_file.set_current_ngram_size(3)
    try:
        loaded_file.set_current_ngram_size(3)
    except IncorrectNgramError:
        loaded_file.set_current_ngram_size(None)

    dynamic_gen = DynamicBackOffGenerator(loaded_file, word_processor)
    print(f'BackOff before:\n{dynamic_gen.run(50, "Ivanov")}')

    loaded_file.update(encoded)
    print(f'BackOff after:\n{dynamic_gen.run(50, "Ivanov")}')

    result = dynamic_gen
    assert result, "Result is None"

if __name__ == "__main__":
    main()
