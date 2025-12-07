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
    processor = WordProcessor('<EoS>')
    encoded_hp = processor.encode_sentences(hp_letters)
    trie = PrefixTrie()
    trie.fill(encoded_hp)
    suggestion = trie.suggest((2,))[0]
    print(f"\nDecoded output: {processor.decode(suggestion)}")
    model = NGramTrieLanguageModel(encoded_hp, 5)
    model.build()
    greedy_before = GreedyTextGenerator(model, processor).run(52, 'Dear')
    beam_before = BeamSearchTextGenerator(model, processor, 3).run('Dear', 52)
    print(f"\nGreedy before: {greedy_before}")
    print(f"Beam before: {beam_before}")
    encoded_ussr = processor.encode_sentences(ussr_letters)
    model.update(encoded_ussr)
    greedy_after = GreedyTextGenerator(model, processor).run(52, 'Dear')
    beam_after = BeamSearchTextGenerator(model, processor, 3).run('Dear', 52)
    print(f"\nGreedy after: {greedy_after}")
    print(f"Beam after: {beam_after}")
    dynamic = DynamicNgramLMTrie(encoded_hp, 5)
    dynamic.build()
    save(dynamic, "./saved_dynamic_trie.json")
    loaded = load("./saved_dynamic_trie.json")
    generator = DynamicBackOffGenerator(loaded, processor)
    print(f"\nDynamic before: {generator.run(50, 'Ivanov')}")
    loaded.update(encoded_ussr)
    size = 3
    max_n = loaded._max_ngram_size 
    if 2 <= size <= max_n:
        loaded.set_current_ngram_size(size)
    else:
        loaded.set_current_ngram_size(max_n)
    print(f"Dynamic after: {generator.run(50, 'Ivanov')}\n")
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
    assert dynamic, "Result is None"

if __name__ == "__main__":
    main()
