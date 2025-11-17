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
    text_processor = TextProcessor(end_of_word_token="_")
    encoded = text_processor.encode(text)
    if encoded is None:
        return
    reader = NGramLanguageModelReader('./assets/en_own.json', '_')
    load_models = []
    for size in range(7):
        load_model = reader.load(size)
        if load_model is not None:
            if size <= 5:
                load_greedy = GreedyTextGenerator(
                    load_model,
                    reader.get_text_processor()
                ).run(20, 'Vermon')
                print(load_greedy)
            if size >= 4:
                load_beam = BeamSearchTextGenerator(
                    load_model,
                    reader.get_text_processor(),
                    2
                ).run('Vermon', 25)
                print(load_beam)
            if size >= 3:
                load_models.append(load_model)
    if load_models:
        load_reader = BackOffGenerator(
            tuple(load_models),
            reader.get_text_processor()
        ).run(30, 'Vernom')
        print(load_reader)
    result = load_reader
    assert result



if __name__ == "__main__":
    main()
