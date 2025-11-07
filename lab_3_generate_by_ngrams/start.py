"""
Generation by NGrams starter
"""

# pylint:disable=unused-import, unused-variable
from lab_3_generate_by_ngrams.main import BeamSearchTextGenerator, GreedyTextGenerator, NGramLanguageModel, TextProcessor


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
    n_gram_model = NGramLanguageModel(encoded_text, 3)
    build_result = n_gram_model.build()
    if build_result == 0:
        n_gram_freq_dict = n_gram_model._n_gram_frequencies
    print(n_gram_freq_dict)
    generator_model = NGramLanguageModel(encoded_text, 7)
    generator_model.build()
    greedy_generator = GreedyTextGenerator(generator_model, processor)
    greedy_algorithm = greedy_generator.run(51, 'Vernon')
    print(greedy_algorithm)
    beam_search_generator = BeamSearchTextGenerator(generator_model, processor, beam_width=3)
    beam_search_result = beam_search_generator.run('Vernon', 56)
    print(beam_search_result)
    result = beam_search_result
    assert result


if __name__ == "__main__":
    main()
