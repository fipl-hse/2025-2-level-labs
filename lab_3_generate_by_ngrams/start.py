"""
Generation by NGrams starter
"""

# pylint:disable=unused-import, unused-variable
from lab_3_generate_by_ngrams.main import (
    BackOffGenerator,
    BeamSearchTextGenerator,
    GreedyTextGenerator,
    NGramLanguageModel,
    TextProcessor,
)


def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    processor = TextProcessor("_")
    encoded_text = processor.encode(text)
    if encoded_text is None:
        return
    processed_text = processor.decode(encoded_text)
    print(processed_text)

    n_gram_model = NGramLanguageModel(encoded_text, 3)
    build_result = n_gram_model.build()
    print(build_result)

    generator_model = NGramLanguageModel(encoded_text, 7)
    generator_model.build()
    greedy_generator = GreedyTextGenerator(generator_model, processor)
    greedy_algorithm = greedy_generator.run(51, 'Vernon')
    print(greedy_algorithm)

    beam_search_generator = BeamSearchTextGenerator(generator_model, processor, beam_width=3)
    beam_search_algorithm = beam_search_generator.run('Vernon', 56)
    print(beam_search_algorithm)

    models = []
    for n_gram_size in [2, 3, 4]:
        model = NGramLanguageModel(encoded_text, n_gram_size)
        model.build()
        models.append(model)
    back_off_generator = BackOffGenerator(tuple(models), processor)
    back_off_algorithm = back_off_generator.run(60, 'Vernon shouted that')
    print(back_off_algorithm)

    result = back_off_algorithm
    assert result


if __name__ == "__main__":
    main()
