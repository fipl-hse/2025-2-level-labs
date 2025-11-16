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
    text_processor = TextProcessor("_")
    encoded_text = text_processor.encode(text)
    if not encoded_text:
        return None
    print(f"\nEncoded Text: {encoded_text}")
    decoded_text = text_processor.decode(encoded_text)
    if not decoded_text:
        return None
    print(f"\nDecoded Text: {decoded_text}")
    language_model=NGramLanguageModel(encoded_text, n_gram_size = 7)
    build = language_model.build()
    if build == 1:
        return None
    greedy_generator = GreedyTextGenerator(language_model, text_processor)
    greedy_text = greedy_generator.run(51, "Vernon")
    if not greedy_text:
        return None
    print(f"\nGreedy text: {greedy_text}")
    beam_generator = BeamSearchTextGenerator(language_model, text_processor, beam_width=3)
    beam_text = beam_generator.run("Vernon", 56)
    if not beam_text:
        return None
    print(f"\nBeam Search Generator: {beam_text}")

    result = beam_text
    assert result


if __name__ == "__main__":
    main()
