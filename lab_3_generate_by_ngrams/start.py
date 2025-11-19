"""
Generation by NGrams starter
"""

# pylint:disable=unused-import, unused-variable
from lab_3_generate_by_ngrams.main import (
    BackOffGenerator,
    BeamSearchTextGenerator,
    GreedyTextGenerator,
    NGramLanguageModel,
    NGramLanguageModelReader,
    TextProcessor,
)


def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    processor = TextProcessor(".")
    encoded_text = processor.encode(text)
    if encoded_text is None:
        return
    model = NGramLanguageModel(encoded_text, 7)
    model.build()
    generator = GreedyTextGenerator(model, processor)
    result_generator = generator.run(51, "Vernon")
    print(result_generator)

    beam_search = BeamSearchTextGenerator(model, processor, 3)
    beam_search_ = beam_search.run("Vernon", 56)
    result_beam = beam_search_
    print(result_beam)

    language_models = []
    for n_gram_size in [1, 2, 3]:
        loaded_model = NGramLanguageModelReader("./assets/en_own.json", "_").load(n_gram_size)
        if loaded_model is not None:
            language_models.append(model)

    back_off = BackOffGenerator(tuple(language_models), processor).run(60, 'Vernon')
    result = back_off
    print(result)
    assert result

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
