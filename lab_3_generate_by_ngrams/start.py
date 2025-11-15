"""
Generation by NGrams starter
"""

# pylint:disable=unused-import, unused-variable
from lab_3_generate_by_ngrams.main import (
    BackOffGenerator,
    BeamSearchTextGenerator,
    GreedyTextGenerator,
    NGramLanguageModel,
    NGramLanguageModelReader,
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
    greedy_algorithm = GreedyTextGenerator(generator_model, processor).run(51, 'Vernon')
    print(greedy_algorithm)

    beam_search_algorithm = BeamSearchTextGenerator(generator_model, processor, 3).run('Vernon', 56)
    print(beam_search_algorithm)

    reader = NGramLanguageModelReader("./assets/en_own.json", "_")
    model_2 = reader.load(2)
    model_3 = reader.load(3)
    model_4 = reader.load(4)
    models = [model_2, model_3, model_4]

    back_off_algorithm = BackOffGenerator(tuple(models), processor).run(60, 'Vernon')
    print(back_off_algorithm)

    result = back_off_algorithm
    assert result


if __name__ == "__main__":
    main()
