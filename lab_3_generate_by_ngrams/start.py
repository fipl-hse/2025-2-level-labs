"""
Generation by NGrams starter
"""

# pylint:disable=unused-import, unused-variable
from lab_3_generate_by_ngrams.main import GreedyTextGenerator, NGramLanguageModel, TextProcessor


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

    n_gram_model = NGramLanguageModel(encoded_text, 5)
    built_freq = n_gram_model.build()
    print(built_freq)

    greedy_text_generator = GreedyTextGenerator(n_gram_model, text_processor)
    greedy_text = greedy_text_generator.run(42, 'Vernon')
    print(greedy_text)

    result = decoded_text
    assert result

if __name__ == "__main__":
    main()
