"""
Generation by NGrams starter
"""

from lab_3_generate_by_ngrams.main import GreedyTextGenerator, NGramLanguageModel, TextProcessor


# pylint:disable=unused-import, unused-variable
def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()

    processor = TextProcessor("_")

    encoded = processor.encode(text)

    if encoded:
        decoded = processor.decode(encoded[:1000])

    prompt = "Vernon"
    n_gram_size = 7
    seq_len = 51

    if encoded:
        model = NGramLanguageModel(encoded, n_gram_size)
        build_result = model.build()

        if build_result == 0:
            generator = GreedyTextGenerator(model, processor)
            result = generator.run(seq_len, prompt)

    result = None
    assert result is None


if __name__ == "__main__":
    main()
