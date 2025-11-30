"""
Generation by NGrams starter
"""

from lab_3_generate_by_ngrams.main import (
    BackOffGenerator,
    BeamSearchTextGenerator,
    GreedyTextGenerator,
    NGramLanguageModel,
    NGramLanguageModelReader,
    TextProcessor,
)


# pylint:disable=unused-import, unused-variable
def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text_content = text_file.read()

    text_processor = TextProcessor("_")
    encoded_content = text_processor.encode(text_content)
    if not encoded_content:
        return

    language_model = NGramLanguageModel(encoded_content, 7)
    language_model.build()

    greedy_generator = GreedyTextGenerator(language_model, text_processor)
    greedy_output = greedy_generator.run(51, "Vernon")
    print(greedy_output)

    beam_generator = BeamSearchTextGenerator(language_model, text_processor, 3)
    beam_output = beam_generator.run("Vernon", 56)
    print(beam_output)

    models_list = []
    for ngram_size in [3, 4, 5]:
        model_loader = NGramLanguageModelReader("./assets/en_own.json", "_")
        loaded_model = model_loader.load(ngram_size)
        if loaded_model is not None:
            models_list.append(loaded_model)

    if models_list:
        backoff_generator = BackOffGenerator(tuple(models_list), text_processor)
        backoff_output = backoff_generator.run(60, 'Vernon')
        print(backoff_output)
        assert backoff_output is not None


if __name__ == "__main__":
    main()
