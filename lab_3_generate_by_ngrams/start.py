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
    processor = TextProcessor(".")
    encoded_text = processor.encode(text)
    if encoded_text is None:
        return
    model = NGramLanguageModel(encoded_text, 7)
    model.build()
    generator = GreedyTextGenerator(model, processor)
    result_generator = generator.run(51, "Vernon")
    beam_search = BeamSearchTextGenerator(model, processor, 3)
    beam_search_ = beam_search.run("Vernon", 56)
    result_beam = beam_search_
    language_models = []
    for n_gram_size in [1, 2, 3]:
        model = NGramLanguageModel(encoded_text, n_gram_size)
        model.build()
        language_models.append(model)
    back_off = BackOffGenerator(language_models, processor)
    result = back_off.run(20, "Vernon")
    print(result_generator)
    print(result_beam)
    print(result)
    assert result


if __name__ == "__main__":
    main()
