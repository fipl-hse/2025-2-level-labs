"""
Generation by NGrams starter
"""

from lab_3_generate_by_ngrams.main import (
    BackOffGenerator,
    BeamSearchTextGenerator,
    GreedyTextGenerator,
    NGramLanguageModel,
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


    text_processor = TextProcessor(end_of_word_token="_")
    encoded_text = text_processor.encode(text)
    if encoded_text is None:
        return


    n_gram_models = {}
    for n_gram_size in range(14):
        n_gram_model = NGramLanguageModel(encoded_text, n_gram_size)
        n_gram_model.build()

        n_gram_models[n_gram_size] = n_gram_model


    greedy_generator = GreedyTextGenerator(n_gram_models[7], text_processor)
    output_greedy: str | None = greedy_generator.run(51, "Vernon")

    beam_generator = BeamSearchTextGenerator(n_gram_models[7], text_processor, 5)
    output_beam: str | None = beam_generator.run("Vernon", 51)

    back_off_generator = BackOffGenerator(tuple(n_gram_models.values()), text_processor)
    output_back_off:  str | None = back_off_generator.run(51, "Vernon")


    print(f"Greedy: {output_greedy}")
    print(f"Beam: {output_beam}")
    print(f"Back off: {output_back_off}")

    result: str | None = output_back_off

    assert result

if __name__ == "__main__":
    main()
