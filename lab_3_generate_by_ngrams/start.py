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
    with open("assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()

    processor = TextProcessor(_end_of_word_token="_")
    encoded_text = processor.encode(text)
    print("Encoded text: ", encoded_text)
    print()
    decoded_text = processor.decode(encoded_text)
    print("Decoded text: ", decoded_text)
    print()
    print("End of word token: ", processor._end_of_word_token)
    n_gram_processor = NGramLanguageModel(encoded_text[:300], 3)

    result = n_gram_processor.build()
    print(result)
    assert result, "Result is None"


if __name__ == "__main__":
    main()