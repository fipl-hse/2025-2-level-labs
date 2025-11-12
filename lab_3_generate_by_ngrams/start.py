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

    processor = TextProcessor ('_')

    tokens = processor._tokenize(text)
    print("Tokens (sample 50):", tokens[:50] if tokens else None)

    encoded_text = processor.encode(text)
    print("Encoded (sample 50):", encoded_text[:50] if encoded_text else None)

    decoded_tokens = processor._decode(encoded_text) if encoded_text else None
    print("Decoded tokens (sample 50):", decoded_tokens[:50] if decoded_tokens else None)

    decoded_text = processor.decode(encoded_text) if encoded_text else None
    print("Decoded text (first 300 characters):", decoded_text[:300] if decoded_text else None)

    assert decoded_text is not None


if __name__ == "__main__":
    main()
