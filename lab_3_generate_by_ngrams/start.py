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


if __name__ == "__main__":
    main()
