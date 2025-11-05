"""
Generation by NGrams starter
"""

# pylint:disable=unused-import, unused-variable
from main import TextProcessor

def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    processor = TextProcessor("_")
    #tokenized_text = processor._tokenize(text)
    encoded_text = processor.encode(text)
    result = processor.decode(encoded_text)
    print(result)
    assert result


if __name__ == "__main__":
    main()
