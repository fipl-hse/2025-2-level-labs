"""
Generation by NGrams starter
"""

# pylint:disable=unused-import, unused-variable
from lab_3_generate_by_ngrams.main import (
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
    print(encoded_text)
    decoded_text = processor.decode(encoded_text) or ()
    print(decoded_text)
    model = NGramLanguageModel(encoded_text, 7)
    model.build()
    greedy_generator = GreedyTextGenerator(model, processor)
    greedy_text = greedy_generator.run(51, 'Vernon')
    print(greedy_text)
    result = greedy_text
    assert result


if __name__ == "__main__":
    main()
