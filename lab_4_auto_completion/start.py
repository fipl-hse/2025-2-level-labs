"""
Auto-completion start
"""

# pylint:disable=unused-variable
from lab_4_auto_completion.main import (
    DynamicBackOffGenerator,
    DynamicNgramLMTrie,
    NGramTrieLanguageModel,
    PrefixTrie,
    WordProcessor,
    load,
    save
)

from lab_3_generate_by_ngrams.main import (
    BeamSearcher,
    BeamSearchTextGenerator,
    GreedyTextGenerator
)


def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/hp_letters.txt", "r", encoding="utf-8") as letters_file:
        hp_letters = letters_file.read()
    with open("./assets/ussr_letters.txt", "r", encoding="utf-8") as text_file:
        ussr_letters = text_file.read()
    with open("./assets/secrets/secret_5.txt", "r", encoding="utf-8") as text_file:
        secret = text_file.read()
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as harry_file:
        text = harry_file.read()
    processor = WordProcessor(end_of_sentence_token="<EOS>")
    encoded_data = processor.encode_sentences(hp_letters)
    words_combined = []
    for sent in encoded_data:
        words_combined.extend(sent)
    tri_grams_tuple = tuple(
        tuple(words_combined[idx:idx + 3])
        for idx in range(len(words_combined) - 2)
    )
    tree = PrefixTrie()
    tree.fill(tri_grams_tuple)
    found = tree.suggest((2,))
    print(f"Found {len(found)} suggestions for prefix (2,)")
    if found:
        best = found[0]
        print(f"First suggestion: {best}")
        output_words = []
        for code in best:
            for text, num in processor._storage.items():
                if num == code:
                    output_words.append(text)
                    break
        decoded_text = processor._postprocess_decoded_text(tuple(output_words))
        print(f"Decoded result: {decoded_text}")
    encoded_hp = processor.encode_sentences(hp_letters)
    n_gram_size = 5
    model = NGramTrieLanguageModel(encoded_hp, n_gram_size)
    greedy_generator = GreedyTextGenerator(model, processor)
    greedy_result_before = greedy_generator.run(seq_len=30, prompt="Ivanov")
    print(f"Result Greedy: {greedy_result_before}")
    beam_searcher = BeamSearcher(3, 10)
    beam_generator = BeamSearchTextGenerator(model, processor, beam_searcher)
    beam_result_before = beam_generator.run(seq_len=30, prompt="Ivanov")
    print(f"Result BeamSearch: {beam_result_before}")
    encoded_ussr = processor.encode_sentences(ussr_letters)
    model.update(encoded_ussr)
    greedy_result_after = greedy_generator.run(seq_len=30, prompt="Ivanov")
    print(f"Result Greedy update: {greedy_result_after}")
    beam_result_after = beam_generator.run(seq_len=30, prompt="Ivanov")
    print(f"Result BeamSearch update: {beam_result_after}")
    dynamic_model = DynamicNgramLMTrie(encoded_hp, n_gram_size=5)
    dynamic_build_result = dynamic_model.build()
    save_path = "./dynamic_model.json"
    save(dynamic_model, save_path)
    loaded_model = load(save_path)
    dynamic_generator = DynamicBackOffGenerator(dynamic_model, processor)
    dynamic_result_before = dynamic_generator.run(seq_len=50, prompt="Ivanov")
    print(f"Result BackOff (before):\n{dynamic_result_before}")
    dynamic_model.update(encoded_ussr)
    dynamic_generator_updated = DynamicBackOffGenerator(dynamic_model, processor)
    dynamic_result_after = dynamic_generator_updated.run(seq_len=50, prompt="Ivanov")
    print(f"Result BackOff (after):\n{dynamic_result_after}")
    result = dynamic_result_after
    assert result, "Result is None"


if __name__ == "__main__":
    main()
