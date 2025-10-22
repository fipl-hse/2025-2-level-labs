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
    with open("assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()

    processor = TextProcessor(end_of_word_token="_")
    tokenized_text = processor._tokenize(text)
    for token in tokenized_text:
        storage = processor._put(token)
    result = storage
    print("_storage =", processor._storage)
    print(result)
    assert result, "Result is None"


if __name__ == "__main__":
    main()