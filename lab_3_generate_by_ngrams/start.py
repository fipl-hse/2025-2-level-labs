"""
Generation by NGrams starter
"""

from main import (
    BackOffGenerator,
    BeamSearchTextGenerator,
    GreedyTextGenerator,
    NGramLanguageModel,
    TextProcessor,
)

# pylint:disable=unused-import, unused-variable


def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("lab_3_generate_by_ngrams\\assets\Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    processor = TextProcessor('_')
    encoded_text = processor.encode(text) or ()
    decoded_text = processor.decode(encoded_text) or ()
    model = NGramLanguageModel(encoded_text, 7)
    model.build()
    greedy_generator = GreedyTextGenerator(model, processor)
    greedy_text = greedy_generator.run(51, 'Vernon')
    print(greedy_text)
    beam_search_generator = BeamSearchTextGenerator(model, processor, 3)
    beam_search_text = beam_search_generator.run('Vernon', 56)
    print(beam_search_text)
    models = []
    for ngram_size in (4, 5, 6):
        model = NGramLanguageModel(encoded_text, ngram_size)
        model.build()
        models.append(model)
    back_off_generator = BackOffGenerator(tuple(models), processor)
    result_back_off = back_off_generator.run(55, 'Vernon')
    print(result_back_off)
    result = result_back_off
    assert result


if __name__ == "__main__":
    main()
