"""
Generation by NGrams starter
"""

# pylint:disable=unused-import, unused-variable

from main import GreedyTextGenerator, NGramLanguageModel, TextProcessor


def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    processor = TextProcessor("_")
    encoded_text = processor.encode(text) or ()
    decoded_text = processor.decode(encoded_text)
    language_model = NGramLanguageModel(encoded_text, 7)
    language_model.build()
    greedy_generator = GreedyTextGenerator(language_model, processor)
    result = greedy_generator.run(51, "Vernon")
    print(result)
    assert result


if __name__ == "__main__":
    main()
