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
    reader = NGramLanguageModelReader('./assets/en_own.json', '_')
    loaded_processor = reader.get_text_processor()
    if encoded:
        custom_model = NGramLanguageModel(encoded, 4)
        custom_model.build()
        custom_greedy = GreedyTextGenerator(
            custom_model,
            processor
        )
        custom_result = custom_greedy.run(15, 'Harry')
        print(f"\nCustom model from Harry Potter: {custom_result}")
    for n_size in (2, 3, 4, 5, 6):
        loaded_model = reader.load(n_size)
        if loaded_model is not None:
            if n_size <= 5:
                greedy_loaded = GreedyTextGenerator(
                    loaded_model,
                    loaded_processor
                )
                result = greedy_loaded.run(20, 'Harry')
                print(f"\nGreedy with loaded {n_size}-gram: {result}")
            if n_size >= 4:
                beam_loaded = BeamSearchTextGenerator(
                    loaded_model,
                    loaded_processor,
                    2
                )
                result = beam_loaded.run('Hermione', 25)
                print(f"\nBeam Search with loaded {n_size}-gram: {result}")
    loaded_model_list = []
    for n_size in (3, 4, 5, 6):
        loaded_model = reader.load(n_size)
        if loaded_model is not None:
            loaded_model_list.append(loaded_model)
    if loaded_model_list:
        backoff_loaded = BackOffGenerator(
            tuple(loaded_model_list),
            loaded_processor
        )
        result = backoff_loaded.run(30, 'Dumbledore')
        print(f"\nBackOff with loaded models: {result}")
        assert result


if __name__ == "__main__":
    main()
