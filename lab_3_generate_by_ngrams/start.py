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
    if encoded_text is None:
        return
    decoded_text = text_processor.decode(encoded_text)
    print(decoded_text)
    n_gram_size = 7
    language_model = NGramLanguageModel(encoded_text, n_gram_size)
    language_model.build()
    greedy_generator = GreedyTextGenerator(language_model, text_processor)
    greedy_text = greedy_generator.run(seq_len=51, prompt="Vernon")
    print(greedy_text)
    beam_search_generator = BeamSearchTextGenerator(language_model, text_processor, beam_width=3)
    beam_text = beam_search_generator.run(prompt="Vernon", seq_len=56)
    print(beam_text)
    result = beam_text
    assert result


if __name__ == "__main__":
    main()
