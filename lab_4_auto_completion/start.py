"""
Auto-completion start
"""

# pylint:disable=unused-variable
from lab_3_generate_by_ngrams.main import BeamSearcher, BeamSearchTextGenerator, GreedyTextGenerator
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

    word_processor = WordProcessor(end_of_sentence_token = '<EoW>')
    encoded_hp = word_processor.encode_sentences(hp_letters)
    tree = PrefixTrie()
    tree.fill(encoded_hp)

    if (found := tree.suggest((2,))):
        best = found[0]
        output_words = [next(text for text, num in word_processor._storage.items() if num == code)
                       for code in best]
        decoded = word_processor._postprocess_decoded_text(tuple(output_words))
        print(f"Found {len(found)} suggestions, first: {decoded}")
    else:
        print('No continuations found')

    model = NGramTrieLanguageModel(encoded_hp, 5)
    model.build()

    greedy = GreedyTextGenerator(model, word_processor)
    beam = BeamSearchTextGenerator(model, word_processor, 3)

    greedy_before = greedy.run(30, "Ivanov")
    beam_before = beam.run("Ivanov", 30)
    print(f'Greedy: {greedy_before}\nBeam: {beam_before}')

    model.update(word_processor.encode_sentences(ussr_letters))

    greedy_after = greedy.run(30, "Ivanov")
    beam_after = beam.run("Ivanov", 30)
    print(f'Greedy update: {greedy_after}\nBeam update: {beam_after}')

    dynamic_model = DynamicNgramLMTrie(encoded_hp, 5)
    dynamic_model.build()

    save(dynamic_model, "./dynamic_model.json")
    load("./dynamic_model.json")

    dynamic_gen = DynamicBackOffGenerator(dynamic_model, word_processor)
    dynamic = dynamic_gen.run(50, "Ivanov")
    print(f'BackOff before:\n{dynamic}')

    dynamic_model.update(word_processor.encode_sentences(ussr_letters))
    dynamic_after = dynamic_gen.run(50, "Ivanov")
    print(f'BackOff after:\n{dynamic_after}')

    result = dynamic_after
    assert result, "Result is None"

if __name__ == "__main__":
    main()
