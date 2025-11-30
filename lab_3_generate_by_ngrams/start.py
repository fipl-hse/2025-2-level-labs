"""
Generation by NGrams starter
"""

from lab_3_generate_by_ngrams.main import (
    BackOffGenerator,
    BeamSearchTextGenerator,
    GreedyTextGenerator,
    NGramLanguageModel,
    NGramLanguageModelReader,
    TextProcessor,
)


# pylint:disable=unused-import, unused-variable
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

    model = NGramLanguageModel(encoded_text, 7)
    model.build()

    generator = GreedyTextGenerator(model, processor)
    result_generator = generator.run(51, "Vernon")
    print(result_generator)

    beam_search = BeamSearchTextGenerator(model, processor, 3)
    beam_search_result = beam_search.run("Vernon", 56)
    print(beam_search_result)

    language_models = []
    for n_gram_size in [3, 4, 5]:
        loaded_model = NGramLanguageModelReader("./assets/en_own.json", "_").load(
            n_gram_size
        )
        if loaded_model is not None:
            language_models.append(loaded_model)

    if language_models:
        back_off = BackOffGenerator(tuple(language_models), processor)
        back_off_result = back_off.run(60, "Vernon")
        print(back_off_result)
        assert back_off_result is not None


if __name__ == "__main__":
    main()
