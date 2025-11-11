"""
Generation by NGrams starter
"""

# pylint:disable=unused-import, unused-variable
from main import BeamSearchTextGenerator, GreedyTextGenerator, NGramLanguageModel, TextProcessor


def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()

    processor = TextProcessor(end_of_word_token="_")
    encoded_text = processor.encode(text)
    print("Encoded text: ", encoded_text)
    print()
    decoded_text = processor.decode(encoded_text)
    print("Decoded text: ", decoded_text)
    print()
    print("End of word token: ", processor._end_of_word_token)
    n_gram_processor = NGramLanguageModel(encoded_text[:300], 7)

    print("N_gram building dict return: ", n_gram_processor.build())

    greedy_text_processor = GreedyTextGenerator(n_gram_processor, processor)
    test_sequense = greedy_text_processor.run(51, "Potter")

    print("Run GreedyTextGenerator for 'Potter', 51: ", test_sequense)

    beam_language_model = NGramLanguageModel(encoded_text[:300], 7)
    beam_language_model.build()
    beam_search_processor = BeamSearchTextGenerator(beam_language_model, processor, 7)
    test_sequense = beam_search_processor.run("Potter", 51)

    print("Run BeamSearchTextGenerator for 'Potter', 51: ", test_sequense)
    result = test_sequense
    assert result, "Result is None"


if __name__ == "__main__":
    main()