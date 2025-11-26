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
    processor = TextProcessor(end_of_word_token='_')
    encoded = processor.encode(text) or tuple()
    if encoded:
        loaded_model_list = []
        for n_size in range(2, 11):
            custom_model = NGramLanguageModel(encoded, n_size)
            custom_model.build()
            loaded_model_list.append(custom_model)
            result = GreedyTextGenerator(
                custom_model,
                processor
            ).run(30, 'Harry')
            print(f"\nGreedy with custom {n_size}-gram: {result}")
            result = BeamSearchTextGenerator(
                custom_model,
                processor,
                2
            ).run('Harry', 30)
            print(f"\nBeam Search with custom {n_size}-gram: {result}")
        if loaded_model_list:
            result = BackOffGenerator(
                tuple(loaded_model_list),
                processor
            ).run(35, 'Harry')
            print(f"\nBackOff with all custom models: {result}")
            assert result


if __name__ == "__main__":
    main()
