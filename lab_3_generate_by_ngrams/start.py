"""
Generation by NGrams starter
"""

# pylint:disable=unused-import, unused-variable
from main import (
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
    processor = TextProcessor(end_of_word_token='_')
    encoded_text = processor.encode(text)
    if not isinstance(encoded_text, tuple) and not encoded_text:
        return
    print(encoded_text)
    decoded_text = processor.decode(encoded_text)
    print(decoded_text)
    model = NGramLanguageModel(encoded_text, 7)
    frequency = model.build()
    print(frequency)
    greedy_generator = GreedyTextGenerator(model, processor)
    result_greedy_generator = greedy_generator.run(51, "Vernon")
    print(result_greedy_generator)
    result = result_greedy_generator
    assert result


if __name__ == "__main__":
    main()
