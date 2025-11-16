"""
Generation by NGrams starter
"""

# pylint:disable=unused-import, unused-variable
from lab_3_generate_by_ngrams.main import (
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
    processor = TextProcessor('_')
    encoded_text = processor.encode(text) or ()

    decoded_text = processor.decode(encoded_text) or ()
    print(decoded_text)

    model = NGramLanguageModel(encoded_text, 7)
    frequency = model.build()
    print(frequency)

    greedy_generator = GreedyTextGenerator(model, processor)
    result_greedy_generator = greedy_generator.run(51, "Vernon")
    print(result_greedy_generator)

    beam_search_generator = BeamSearchTextGenerator(model, processor, 3)
    result_beam_search = beam_search_generator.run('Vernon', 56)
    print(result_beam_search)

    result = result_beam_search
    assert result


if __name__ == "__main__":
    main()
