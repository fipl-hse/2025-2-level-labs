"""
Generation by NGrams starter
"""

from lab_3_generate_by_ngrams.main import (
    GreedyTextGenerator,
    NGramLanguageModel,
    TextProcessor,
    BeamSearchTextGenerator,
    BackOffGenerator,
    NGramLanguageModelReader
)


# pylint:disable=unused-import, unused-variable
def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as file:
        content = file.read()

    text_handler = TextProcessor("_")
    processed_text = text_handler.encode(content)
    if processed_text is None:
        return

    ngram_model = NGramLanguageModel(processed_text, 7)
    ngram_model.build()
    
    greedy_engine = GreedyTextGenerator(ngram_model, text_handler)
    greedy_text = greedy_engine.run(51, "Vernon")
    print(greedy_text)

    search_engine = BeamSearchTextGenerator(ngram_model, text_handler, 3)
    beam_text = search_engine.run("Vernon", 56)
    print(beam_text)

    models_collection = []
    for gram_size in [4, 5, 6]:
        loaded_gram_model = NGramLanguageModelReader("./assets/contexts.json", "_").load(gram_size)
        if loaded_gram_model is not None:
            models_collection.append(loaded_gram_model)

    backoff_engine = BackOffGenerator(tuple(models_collection), text_handler)
    backoff_text = backoff_engine.run(60, 'Vernon')
    print(backoff_text)
    assert backoff_text is not None


if __name__ == "__main__":
    main()
