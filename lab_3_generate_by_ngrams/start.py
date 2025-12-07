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
    result = None
    text_processor =  TextProcessor(end_of_word_token = '_')
    encoded_text = text_processor.encode(text)
    print(encoded_text)
    if encoded_text is None or None in encoded_text:
        return
    decoded_text = text_processor.decode(encoded_text)
    print(decoded_text)
    n_gram_model = NGramLanguageModel(encoded_text, 7)
    n_gram_model.build()
    built_frequency = n_gram_model.build()
    print(built_frequency)
    greedy_generator = GreedyTextGenerator(n_gram_model, text_processor)
    greedy_text = greedy_generator.run(51, 'Vernon')
    print(greedy_text)
    beam_generator = BeamSearchTextGenerator(n_gram_model, text_processor, 7)
    result = beam_generator.run("Vernon", 56)
    print(result)
    assert result
    return


if __name__ == "__main__":
    main()
