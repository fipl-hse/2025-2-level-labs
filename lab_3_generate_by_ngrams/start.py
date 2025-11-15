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
    for n_size in (2, 3):
        model_n = reader.load(n_size)
        if model_n:
            greedy_result = GreedyTextGenerator(model_n, reader.get_text_processor()).run(10, 'Vernon')
            print(f"Loaded model {n_size}-gram: {greedy_result}")
    model = NGramLanguageModel(encoded, 7)
    model.build()
    print(f'\nGreedy: {GreedyTextGenerator(model, processor).run(51, "Vernon")}')
    beam_result = BeamSearchTextGenerator(model, processor, 3).run('Vernon', 56)
    print(f'\nBeam Search: {beam_result}')
    models = []
    for ngram_size in (4, 5, 6):
        current_model = NGramLanguageModel(encoded, ngram_size)
        current_model.build()
        models.append(current_model)
    backoff_result = BackOffGenerator(tuple(models), processor).run(55, 'Vernon')
    print(f'\nBackOff: {backoff_result}')
    result = backoff_result
    assert result


if __name__ == "__main__":
    main()
