"""
Generation by NGrams starter
"""

# pylint:disable=unused-import, unused-variable
from lab_3_generate_by_ngrams.main import TextProcessor

def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    processor = TextProcessor("_")
    encoded_text = processor.encode(text)
    processed_text = processor.decode(encoded_text)
    print(processed_text)
    result = processed_text
    assert result


if __name__ == "__main__":
    main()
