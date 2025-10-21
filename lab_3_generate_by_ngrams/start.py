"""
Generation by NGrams starter
"""
from lab_3_generate_by_ngrams.main import (
    TextProcessor,
    NGramLanguageModel,
)
# pylint:disable=unused-import, unused-variable


def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()

    text_test = (
            "She is happy. He is happy"
        )
    # Шаг 1.10. Продемонстрировать результаты в start.py

    text_test_numbers = "123456"
    t = ('s', 'h', 'e', '_', 'i', 's', '_', 'h', 'a', 'p', 'p', 'y', '_',
        'h', 'e', '_', 'i', 's', '_', 'h', 'a', 'p', 'p', 'y', '_')

    encoded = (1, 2, 3, 0, 4, 1, 0, 2, 5, 6, 6, 7, 0, 2, 3, 0, 4, 1, 0, 2, 5, 6, 6, 7, 0)

    ngram_model = NGramLanguageModel(None, 20)

    # processor = TextProcessor("_")
    # proocessed_text = processor._postprocess_decoded_text(t)
    # print(proocessed_text)

    extracted_n_grams = ngram_model._extract_n_grams(encoded_corpus=encoded)
    print(extracted_n_grams)

    # tokens = processor._tokenize(text_test)

    # result = None
    # assert result


if __name__ == "__main__":
    main()
