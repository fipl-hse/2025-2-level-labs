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
    model = NGramLanguageModel(encoded, 7)
    model.build()
    models = []
    for ngram_size in range(7):
        ngram_model = NGramLanguageModel(encoded, ngram_size)
        ngram_model.build()
        models.append(ngram_model)
    greedy_generator = GreedyTextGenerator(model, text_processor)
    beam_generator = BeamSearchTextGenerator(model, text_processor, 5)
    back_off_generator = BackOffGenerator(tuple(models), text_processor)

    print("greedy", greedy_generator.run(51, "Vernon"))
    print("beam", beam_generator.run("Vernon", 56))
    print("back off", back_off_generator.run(51, "Vernon"))
    reader = NGramLanguageModelReader('./assets/en_own.json', '_')
    models = []
    for size in range(7):
        model = reader.load(size)
        if model is not None:
            models.append(model)
    if models:
        reader = BackOffGenerator(tuple(models), reader.get_text_processor())
        load_reader = reader.run(44, 'Vernon')
        print("load", load_reader)
    result = load_reader
    assert result



if __name__ == "__main__":
    main()
