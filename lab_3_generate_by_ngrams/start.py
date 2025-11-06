"""
Generation by NGrams starter
"""

# pylint:disable=unused-import, unused-variable
from lab_3_generate_by_ngrams.main import (TextProcessor)


def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    identif = TextProcessor("_")
    encoded_text=identif.encode(text)
    print(f"Encoded text:{encoded_text}")
    decoded_text = identif.decode (encoded_text)
    print(f"Decoded text: {decoded_text}")
    result = encoded_text, decoded_text
    assert result


if __name__ == "__main__":
    main()
