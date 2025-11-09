"""
Generation by NGrams starter
"""

# pylint:disable=unused-import, unused-variable
from lab_3_generate_by_ngrams.main import (
    BeamSearchTextGenerator,
    GreedyTextGenerator,
    NGramLanguageModel,
    TextProcessor,
)


def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    processor = TextProcessor(end_of_word_token='_')
    encoded_text = processor.encode(text)
    if not isinstance(encoded_text, tuple) and not encoded_text:
        return
    print(encoded_text)
    decoded_text = processor.decode(encoded_text)
    print(decoded_text)
    ngram_model = NGramLanguageModel(encoded_text[:100], n_gram_size=3)
    model_test = NGramLanguageModel(encoded_text, 7)
    greedy_text_generator = GreedyTextGenerator(model_test, processor)
    print(greedy_text_generator.run(51, 'Vernon'))
    beam_search_generator = BeamSearchTextGenerator(model_test, processor, 7)
    print(beam_search_generator.run('Vernon', 56))
    result = beam_search_generator
    assert result

if __name__ == "__main__":
    main()
