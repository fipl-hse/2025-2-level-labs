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
    encoded_text = processor.encode(text) or tuple()

    #decoded_text = processor.decode(encoded_text) or tuple()

    reader = NGramLanguageModelReader('./assets/en_own.json', '_')

    for n_size in (2, 3):
        loaded_model = reader.load(n_size)
        if loaded_model:
            result = GreedyTextGenerator(loaded_model,
                                         reader.get_text_processor()).run(10, 'Vernon')
            print(f"Загруженная модель {n_size}-gram: {result}")

    model_with_7 = NGramLanguageModel(encoded_text, 7)
    model_with_7.build()

    print(f'\n Greedy Algorithm: {GreedyTextGenerator(model_with_7, processor).run(51, 'Vernon')}')

    beam_search_generator = BeamSearchTextGenerator(model_with_7, processor, 3).run('Vernon', 56)
    print(f'\n Beam Search Algorithm: {beam_search_generator}')

    models = []
    for ngram_size in (4, 5, 6):
        model = NGramLanguageModel(encoded_text, ngram_size)
        model.build()
        models.append(model)
    back_off_generator = BackOffGenerator(tuple(models), processor).run(55, 'Vernon')
    print(f'\n BackOff Algorithm: {back_off_generator}\n')

    result = back_off_generator
    assert result

if __name__ == "__main__":
    main()
