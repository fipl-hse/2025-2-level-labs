"""
Generation by NGrams starter
"""

# pylint:disable=unused-import, unused-variable
from lab_3_generate_by_ngrams.main import (
    BackOffGenerator,
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
    result = None
    text_processor = TextProcessor('_')
    encoded = text_processor.encode(text) or ()
    decoded = text_processor.decode(encoded) or ()
    model = NGramLanguageModel(encoded, 7)
    model.build()
    greedy_text_generator = GreedyTextGenerator(model, text_processor)
    greedy_text = greedy_text_generator.run(51, 'Vernon')
    print(greedy_text)
    beam_search_generator = BeamSearchTextGenerator(model, text_processor, 3)
    beam_search_text = beam_search_generator.run('Vernon', 56)
    print(beam_search_text)
    models = []
    for ngram_size in (4, 5, 6):
        model = NGramLanguageModel(encoded, ngram_size)
        model.build()
        models.append(model)
    backoff_generator = BackOffGenerator(tuple(models), text_processor)
    result_backoff = backoff_generator.run(55, 'Vernon')
    print(result_backoff)
    result = result_backoff
    assert result


if __name__ == "__main__":
    main()
