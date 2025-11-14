"""
Generation by NGrams starter
"""

# pylint:disable=unused-import, unused-variable

from main import NGramLanguageModel, TextProcessor


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
    n_gram_language_model = NGramLanguageModel(encoded_text, 3)
    build_result = n_gram_language_model.build()
    result = build_result
    assert result


if __name__ == "__main__":
    main()
