"""
Generation by NGrams starter
"""

# pylint:disable=unused-import, unused-variable
from lab_3_generate_by_ngrams.main import (
    BackOffGenerator,
    BeamSearcher,
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
    processor = TextProcessor(end_of_word_token='_')
    encoded_content = processor.encode(text) or tuple()
    reader = NGramLanguageModelReader('./assets/en_own.json', '_')
    for n_size in (2, 3):
        loaded_model = reader.load(n_size)
        if loaded_model:
            result = GreedyTextGenerator(
                loaded_model,
                reader.get_text_processor()
                ).run(10, 'Vernon')
            print(f"Loaded model {n_size}-gram: {result}")
    model = NGramLanguageModel(encoded_content, 7)
    model.build()
    build_result = model.build()
    print(f"\nBuilt model: {build_result}")
    n_gram_size = model.get_n_gram_size()
    print(f"\nN-gram size: {n_gram_size}")
    test_sequence = (7, 5, 6, 6)
    next_token = model.generate_next_token(test_sequence)
    print(f"\nNext token for {test_sequence}: {next_token}")
    greedy = GreedyTextGenerator(model, processor)
    greedy_result = greedy.run(51, 'Vernon')
    print(f"\nGreedy: {greedy_result}")
    beam_searcher = BeamSearcher(beam_width=3, language_model=model)
    beamsearch = BeamSearchTextGenerator(model, processor, 3)
    beamsearch_result = beamsearch.run('Vernon', 56)
    print(f"\nBeam Search: {beamsearch_result}")
    test_frequencies = {
        (1, 2, 3): 1.0,
        (2, 3, 4): 0.8,
        (3, 4, 5): 0.6
        }
    model.set_n_grams(test_frequencies)
    test_seq = (1, 2, 3, 0, 4, 1)
    next_tokens = beam_searcher.get_next_token(test_seq)
    print(f"\nBeamSearcher next tokens for {test_seq}: {next_tokens}")
    models = []
    for ngram_size in (4, 5, 6):
        current_model = NGramLanguageModel(encoded_content, ngram_size)
        current_model.build()
        models.append(current_model)
    backoff_generator = BackOffGenerator(tuple(models), processor)
    backoff_result = backoff_generator.run(55, 'Vernon')
    print(f"\nBackOff Algorithm: {backoff_result}")
    result = backoff_result
    assert result


if __name__ == "__main__":
    main()
