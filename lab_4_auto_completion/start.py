"""
Auto-completion start
"""

# pylint:disable=unused-variable

from lab_3_generate_by_ngrams.main import (
    BeamSearchTextGenerator,
    GreedyTextGenerator,
)
from lab_4_auto_completion.main import (
    DynamicBackOffGenerator,
    DynamicNgramLMTrie,
    load,
    NGramTrieLanguageModel,
    PrefixTrie,
    save,
    WordProcessor,
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
    text_processor = WordProcessor('<EoS>')
    encoded_hp = text_processor.encode_sentences(hp_letters)
    prefix_tree = PrefixTrie()
    prefix_tree.fill(encoded_hp)
    recommended = prefix_tree.suggest((2,))[0]
    print(f"\nDecoded output: {text_processor.decode(recommended)}")
    language_model = NGramTrieLanguageModel(encoded_hp, 5)
    language_model.build()
    greedy_gen = GreedyTextGenerator(language_model, text_processor)
    beam_gen = BeamSearchTextGenerator(language_model, text_processor, 3)
    print(f"\nGreedy output prior update: {greedy_gen.run(52, 'Dear')}")
    print(f"Beam output prior update: {beam_gen.run('Dear', 52)}")
    encoded_ussr = text_processor.encode_sentences(ussr_letters)
    language_model.update(encoded_ussr)
    print(f"\nGreedy output after update: {greedy_gen.run(52, 'Dear')}")
    beam_output_updated = BeamSearchTextGenerator(language_model, text_processor, 3).run('Dear', 52)
    print(f"Beam output after update: {beam_output_updated}")
    dynamic_model = DynamicNgramLMTrie(encoded_hp, 5)
    dynamic_model.build()
    save(dynamic_model, "./saved_dynamic_trie.json")
    restored_trie = load("./saved_dynamic_trie.json")
    dynamic_generator = DynamicBackOffGenerator(restored_trie, text_processor)
    print(f"\nDynamic output prior update: {dynamic_generator.run(50, 'Ivanov')}")
    restored_trie.update(encoded_ussr)
    target_size = 3
    if target_size >= 2 and target_size <= restored_trie._max_ngram_size:
        restored_trie.set_current_ngram_size(target_size)
    else:
        restored_trie.set_current_ngram_size(restored_trie._max_ngram_size)
    print(f"Dynamic output after update: {dynamic_generator.run(50, 'Ivanov')}\n")
    final_result = dynamic_generator
    # print('\nSolution of the 2nd secret')
    # n_size = 4
    # beam_size = 3
    # generate_length = 25
    # with open("./assets/secrets/secret_3.txt", "r", encoding="utf-8") as file:
    #     encrypted_text = file.read()
    # with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
    #     harry_potter_text = text_file.read()
    # sentences_encoded = text_processor.encode_sentences(harry_potter_text)
    # all_tokens = []
    # for sent in sentences_encoded:
    #     all_tokens.extend(sent)
    # corpus_tokens = tuple(all_tokens)
    # model = NGramLanguageModel(corpus_tokens, n_size)
    # model.build()
    # text_parts = encrypted_text.split("<BURNED>")
    # beginning = text_parts[0].strip()
    # beginning_encoded = processor.encode_sentences(beginning)
    # initial_context = []
    # for sentence in beginning_encoded:
    #     for token_id in sentence:
    #         word = processor.get_token(token_id)
    #         if word != '<EOS>':
    #             initial_context.append(token_id)
    # current_context = tuple(initial_context)
    # search_engine = BeamSearcher(beam_size, model)
    # possible_sequences = {current_context: 0.0}
    # for step in range(generate_length):
    #     updated_sequences = {}
    #     for sequence, probability in possible_sequences.items():
    #         possible_next = search_engine.get_next_token(sequence)
    #         if not possible_next:
    #             continue
    #         extended = search_engine.continue_sequence(
    #             sequence, possible_next, {sequence: probability}
    #         )
    #         if not extended:
    #             continue
    #         for extended_seq, extended_prob in extended.items():
    #             if (
    #                 extended_seq not in updated_sequences or 
    #                 extended_prob < updated_sequences[extended_seq]
    #             ):
    #                 updated_sequences[extended_seq] = extended_prob
    #     if not updated_sequences:
    #         break
    #     filtered = search_engine.prune_sequence_candidates(updated_sequences)
    #     possible_sequences = filtered or {}
    #     if not possible_sequences:
    #         break
    # if possible_sequences:
    #     optimal_sequence = min(possible_sequences.items(), key=lambda x: x[1])[0]
    #     context_length = len(current_context)
    #     predicted_tokens = []
    #     for token_id in optimal_sequence[context_length:]:
    #         word = processor.get_token(token_id)
    #         if word and word != '<EOS>':
    #             predicted_tokens.append(word)
    #     if predicted_tokens:
    #         missing_section = " ".join(predicted_tokens)
    #         split_words = missing_section.split()
    #         if all(len(w) == 1 for w in split_words):
    #             missing_section = missing_section.replace(" ", "")
    #         completed_text = encrypted_text.replace("<BURNED>", missing_section)
    #         print(f"\nComplete letter:\n{completed_text}")
    assert final_result, "Result is None"

if __name__ == "__main__":
    main()
