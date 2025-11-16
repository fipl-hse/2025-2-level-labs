"""
Generation by NGrams starter
"""
from lab_3_generate_by_ngrams.main import (
    BeamSearchTextGenerator,
    GreedyTextGenerator,
    NGramLanguageModel,
    TextProcessor,
)

# pylint:disable=unused-import, unused-variable


def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    result = None
    text_processor = TextProcessor(end_of_word_token = '_')
    encoded_text = text_processor.encode(text)
    if encoded_text is None:
        return

    decoded_text = text_processor.decode(encoded_text) or tuple()
    print('Decoded text: ', decoded_text)
    language_model = NGramLanguageModel(encoded_text, 7)
    builded_model = language_model.build()
    text_generator = GreedyTextGenerator(language_model, text_processor)
    final_text_generator = text_generator.run(51, 'Vernon')
    print('Final text generator: ', final_text_generator)

    beam_search_text = BeamSearchTextGenerator(language_model, text_processor, beam_width = 3)
    final_beam_search_text = beam_search_text.run('Vernon', 56)
    print('Final beam search text: ', final_beam_search_text)
    result = final_beam_search_text
    assert result


if __name__ == "__main__":
    main()
