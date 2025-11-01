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
    processor = TextProcessor('_')
    
    # Encode the text
    encoded_text = processor.encode(text)
    print(f"Encoded text length: {len(encoded_text) if encoded_text else 0}")
    
    # Decode the text
    decoded_text = processor.decode(encoded_text) if encoded_text else None
    print(f"Decoded text sample: {decoded_text[:100] if decoded_text else None}")
    
    # Demonstrate encoding and decoding with a small sample
    sample_text = "She is happy. He is happy."
    sample_encoded = processor.encode(sample_text)
    sample_decoded = processor.decode(sample_encoded) if sample_encoded else None
    
    print(f"Original sample: {sample_text}")
    print(f"Encoded sample: {sample_encoded}")
    print(f"Decoded sample: {sample_decoded}")    
    result = sample_decoded
    assert result


if __name__ == "__main__":
    main()
