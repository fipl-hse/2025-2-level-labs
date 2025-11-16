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

    text_processor = TextProcessor(end_of_word_token='_')

    encoded_content = text_processor.encode(text) or tuple()
    print("Encoded text sample:", encoded_content[:200])

    decoded_content = text_processor.decode(encoded_content) or ""
    print("Decoded text sample:", decoded_content[:200])

    language_model = NGramLanguageModel(encoded_content, 6)
    build_status = language_model.build()
    print(f"Model build status: {build_status}")

    greedy_gen = GreedyTextGenerator(language_model, text_processor)
    greedy_output = greedy_gen.run(31, 'Harry')
    print(f"Greedy generation result: {greedy_output}")

    beam_gen = BeamSearchTextGenerator(language_model, text_processor, 3)
    beam_output = beam_gen.run('Harry', 31)
    print(f"Beam search generation result: {beam_output}")

    result = beam_output
    assert result


if __name__ == "__main__":
    main()
