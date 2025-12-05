"""
Auto-completion start
"""

# pylint:disable=unused-variable
from lab_3_generate_by_ngrams.main import (
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

    word_processor = WordProcessor('<EoW>')

    tokenized_text = word_processor._tokenize(hp_letters)
    encoded_sentences = word_processor.encode_sentences(tokenized_text)
    prefix_trie = PrefixTrie()
    prefix_trie.fill(encoded_sentences)
    suggestion = prefix_trie.suggest((2,))

    if suggestion:
        first_suggestion = suggestion[0]
        decoded_sequence = word_processor._postprocess_decoded_text(first_suggestion)
        print('First sequence:', decoded_sequence)
        result = decoded_sequence
    else:
        print('No continuations were found for the prefix')

    n_gram_size = 5
    encoded_hp = word_processor.encode_sentences(hp_letters)
    model = NGramTrieLanguageModel(encoded_hp, n_gram_size)
    model.build()

    greedy_generator = GreedyTextGenerator(model, word_processor)
    greedy_result_before = greedy_generator.run(seq_len=30, prompt="Ivanov")
    print(f"Result Greedy (before update): {greedy_result_before}")
    
    beam_generator = BeamSearchTextGenerator(model, word_processor)
    beam_result_before = beam_generator.run(seq_len=30, prompt="Ivanov")
    print(f"Result BeamSearch (before update): {beam_result_before}")

    encoded_ussr = word_processor.encode_sentences(ussr_letters)
    model.update(encoded_ussr)

    greedy_result_after = greedy_generator.run(seq_len=30, prompt="Ivanov")
    print(f"Result Greedy (after update): {greedy_result_after}")
    
    beam_result_after = beam_generator.run(seq_len=30, prompt="Ivanov")
    print(f"Result BeamSearch (after update): {beam_result_after}")

    dynamic_model = DynamicNgramLMTrie(encoded_hp, n_gram_size=5)
    dynamic_build_result = dynamic_model.build()
    print(f"Dynamic model built: {dynamic_build_result == 0}")

    save_path = "./dynamic_model.json"
    save(dynamic_model, save_path)
    loaded_model = load(save_path)

    dynamic_generator = DynamicBackOffGenerator(dynamic_model, word_processor)
    dynamic_result_before = dynamic_generator.run(seq_len=50, prompt="Ivanov")
    print(f"Result BackOff (before update):\n{dynamic_result_before}")

    dynamic_model.update(encoded_ussr)

    dynamic_result_after = dynamic_generator.run(seq_len=50, prompt="Ivanov")
    print(f"Result BackOff (after update):\n{dynamic_result_after}")

    result = dynamic_result_after
    assert result, "Result is None"

if __name__ == "__main__":
    main()
