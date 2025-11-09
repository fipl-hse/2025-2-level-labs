"""
Generation by NGrams starter
"""
from lab_3_generate_by_ngrams.main import (
    TextProcessor
)

# pylint:disable=unused-import, unused-variable


def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    
    text_processor =  TextProcessor(end_of_word_token = '_')
    encoded_text = text_processor.encode(text)
    decoded_text = text_processor.decode(encoded_text)

    result = decoded_text
    assert result


if __name__ == "__main__":
    main()
