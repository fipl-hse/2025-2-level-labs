"""
Auto-completion start
"""

# pylint:disable=unused-variable
from lab_3_generate_by_ngrams.main import (
    BeamSearcher,
    BeamSearchTextGenerator,
    GreedyTextGenerator
)
from lab_4_auto_completion.main import (
    DynamicBackOffGenerator,
    DynamicNgramLMTrie,
    NGramTrieLanguageModel,
    PrefixTrie,
    WordProcessor,
    load,
    save
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
    # with open("./assets/secrets/secret_5.txt", "r", encoding="utf-8") as text_file:
    #     secret = text_file.read()
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as harry_file:
        text = harry_file.read()
    processor = WordProcessor(end_of_sentence_token="<EOS>")
    encoded_data = processor.encode_sentences(hp_letters)
    words = []
    for sentence in encoded_data:
        words.extend(sentence)
    tri_grams_tuple = tuple(
        tuple(words[i:i + 3]) for i in range(len(words) - 2))
    tree = PrefixTrie()
    tree.fill(tri_grams_tuple)
    found = tree.suggest((2,))
    if found:
        first = found[0]
        print(f"First: {first}")
        output_words = []
        for i in first:
            for text, j in processor._storage.items():
                if j == i:
                    output_words.append(text)
                    break
        decoded = processor._postprocess_decoded_text(tuple(output_words))
        print(f"Decoded: {decoded}")
    encoded_hp = processor.encode_sentences(hp_letters)
    n_gram_size = 5
    model = NGramTrieLanguageModel(encoded_hp, n_gram_size)
    greedy = GreedyTextGenerator(model, processor)
    greedy_first = greedy.run(seq_len=30, prompt="Ivanov")
    beam_searcher = BeamSearcher(3, 10)
    beam = BeamSearchTextGenerator(model, processor, beam_searcher)
    beam_first = BeamSearchTextGenerator(model, processor, beam_searcher).run(seq_len=30, prompt="Ivanov")
    encoded_ussr = processor.encode_sentences(ussr_letters)
    model.update(encoded_ussr)
    greedy_second = greedy.run(seq_len=30, prompt="Ivanov")
    beam_second = beam.run(seq_len=30, prompt="Ivanov")
    dynamic_model = DynamicNgramLMTrie(encoded_hp, n_gram_size=5)
    dynamic_build = dynamic_model.build()
    path = "./dynamic_model.json"
    save(dynamic_model, path)
    loaded_model = load(path)
    dynamic = DynamicBackOffGenerator(dynamic_model, processor)
    dynamic_first = dynamic.run(seq_len=50, prompt="Ivanov")
    dynamic_model.update(encoded_ussr)
    dynamic_generator_updated = DynamicBackOffGenerator(dynamic_model, processor)
    dynamic_second = dynamic_generator_updated.run(seq_len=50, prompt="Ivanov")
    print(f"Greedy: {greedy_first}")
    print(f"Beam: {beam_first}")
    print(f"Greedy new: {greedy_second}")
    print(f"BeamSearch new: {beam_second}")
    print(f"BackOff: {dynamic_first}")
    print(f"BackOff new: {dynamic_second}")
    result = dynamic_second
    assert result, "Result is None"


if __name__ == "__main__":
    main()
