"""
Generation by NGrams starter
"""

# pylint:disable=unused-import, unused-variable
from lab_3_generate_by_ngrams.main import (
    BeamSearchTextGenerator,
    BeamSearcher,
    GreedyTextGenerator,
    NGramLanguageModel,
    NGramLanguageModelReader,
    BackOffGenerator,
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
    encoded_content = processor.encode(text)
    model = NGramLanguageModel(encoded_content, 7)
    build_result = model.build()
    print(f"\nBuilt model: {build_result}")
    n_gram_size = model.get_n_gram_size()
    print(f"\nN-gram size: {n_gram_size}")
    test_sequence = (7, 5, 6, 6)
    next_token = model.generate_next_token(test_sequence)
    print(f"\nNext token for {test_sequence}: {next_token}")
    greedy = GreedyTextGenerator(model, processor)
    greedy_result = greedy.run(51, 'Vernon')
    print("\nGreedy result:", greedy_result)
    beam_searcher = BeamSearcher(beam_width=3, language_model=model)
    beamsearch = BeamSearchTextGenerator(model, processor, 3)
    beamsearch_result = beamsearch.run('Vernon', 56)
    print("\nBeam Search result:", beamsearch_result)
    test_frequencies = {
        (1, 2, 3): 1.0,
        (2, 3, 4): 0.8,
        (3, 4, 5): 0.6
        }
    set_result = model.set_n_grams(test_frequencies)
    print(f"\nN-gram frequencies: {set_result}")
    reader = NGramLanguageModelReader("./assets/en_own.json", "_")
    models_dict = {}
    text_processor = reader.get_text_processor()
    for n_size in [2, 3, 4]:
        model_n = reader.load(n_size)
        if model_n:
            models_dict[n_size] = model_n
    backoff_generator = BackOffGenerator(list(models_dict.values()), text_processor)
    backoff_result = backoff_generator.run(50, 'Vernon')
    print("\nBackOff generation:", backoff_result)
    test_seq = (1, 2, 3, 0, 4, 1)
    next_tokens = beam_searcher.get_next_token(test_seq)
    print(f"\nBeamSearcher next tokens for {test_seq}: {next_tokens}")
    sequence_candidates = {(1, 2, 3, 0, 4, 1, 0, 2): 0.0}
    next_tokens_list = [(5, 0.6666666666666666), (3, 0.3333333333333333)]
    continue_result = beam_searcher.continue_sequence(
        sequence=(1, 2, 3, 0, 4, 1, 0, 2),
        next_tokens=next_tokens_list,
        sequence_candidates=sequence_candidates
        )
    print(f"\nContinuation of the sequence: {continue_result}")
    test_candidates = {
        (1, 2, 3, 0, 4, 1, 0, 2, 5, 6, 6, 7, 0, 2, 5): 0.8109302162163289,
        (1, 2, 3, 0, 4, 1, 0, 2, 5, 6, 6, 7, 0, 2, 3): 1.5040773967762742,
        (1, 2, 3, 0, 4, 1, 0, 2, 3, 0, 4, 1, 0, 2, 5): 1.5040773967762742,
        (1, 2, 3, 0, 4, 1, 0, 2, 3, 0, 4, 1, 0, 2, 3): 2.1972245773362196
        }
    pruned = beam_searcher.prune_sequence_candidates(test_candidates)
    print(f"\nPruned candidated: {pruned}")


if __name__ == "__main__":
    main()
