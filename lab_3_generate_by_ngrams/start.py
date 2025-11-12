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
    language_model = NGramLanguageModel(encoded_text, 3)
    build_result = language_model.build()
    result = build_result
    print(language_model._n_gram_frequencies)
    assert result


if __name__ == "__main__":
    main()
