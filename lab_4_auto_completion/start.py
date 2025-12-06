"""
Auto-completion start
"""

# pylint:disable=unused-variable

import math

from lab_3_generate_by_ngrams.main import (
    BeamSearchTextGenerator,
    GreedyTextGenerator,
    NGramLanguageModel,
    BeamSearcher
)

from lab_4_auto_completion.main import (
    WordProcessor,
    PrefixTrie,
    NGramTrieLanguageModel,
    DynamicNgramLMTrie,
    DynamicBackOffGenerator,
    save,
    load,
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
    processor = WordProcessor(end_of_sentence_token="<EOS>")
    encoded_sentences = processor.encode_sentences(hp_letters)
    prefix_trie = PrefixTrie()
    prefix_trie.fill(encoded_sentences)
    suggestions = prefix_trie.suggest((2,))
    if suggestions:
        decoded = processor.decode(suggestions[0])
        print(f"Prefix Trie suggestion: {decoded.replace('<EOS>', '').strip()}")
    n_gram_size = 5
    model = NGramTrieLanguageModel(encoded_sentences, n_gram_size)
    model.build()
    greedy_before = GreedyTextGenerator(model, processor)
    gb_result = greedy_before.run(20, 'Dear Harry')
    print(f"Greedy before update: {gb_result}")
    beam_before = BeamSearchTextGenerator(model, processor, 3)
    bb_result = beam_before.run('Dear Harry', 20)
    print(f"Beam before update: {bb_result}")
    encoded_ussr = processor.encode_sentences(ussr_letters)
    if encoded_ussr:
        model.update(encoded_ussr)
    greedy_after = GreedyTextGenerator(model, processor)
    ga_result = greedy_after.run(20, 'Dear Harry')
    print(f"Greedy after update: {ga_result}")
    beam_after = BeamSearchTextGenerator(model, processor, 3)
    ba_result = beam_after.run('Dear Harry', 20)
    print(f"Beam after update: {ba_result}")
    dynamic_model = DynamicNgramLMTrie(encoded_sentences, n_gram_size=5)
    dynamic_model.build()
    save(dynamic_model, "./dynamic_model.json")
    loaded_model = load("./dynamic_model.json")
    dynamic_generator = DynamicBackOffGenerator(loaded_model, processor)
    dynamic_result = dynamic_generator.run(15, "Ivanov")
    print(f"Dynamic generator before update: {dynamic_result}")
    if encoded_ussr:
        loaded_model.update(encoded_ussr)
        dynamic_result_after = dynamic_generator.run(15, "Ivanov")
        print(f"Dynamic generator after update: {dynamic_result_after}")

    print('\nSolution of the 2nd secret')
    n_size = 4
    beam_size = 3
    generate_length = 25
    with open("./assets/secrets/secret_3.txt", "r", encoding="utf-8") as file:
        encrypted_text = file.read()
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        harry_potter_text = text_file.read()
    sentences_encoded = processor.encode_sentences(harry_potter_text)
    all_tokens = []
    for sent in sentences_encoded:
        all_tokens.extend(sent)
    corpus_tokens = tuple(all_tokens)
    model = NGramLanguageModel(corpus_tokens, n_size)
    model.build()
    text_parts = encrypted_text.split("<BURNED>")
    beginning = text_parts[0].strip()
    beginning_encoded = processor.encode_sentences(beginning)
    initial_context = []
    for sentence in beginning_encoded:
        for token_id in sentence:
            word = processor.get_token(token_id)
            if word != '<EOS>':
                initial_context.append(token_id)
    current_context = tuple(initial_context)
    search_engine = BeamSearcher(beam_size, model)
    possible_sequences = {current_context: 0.0}
    for step in range(generate_length):
        updated_sequences = {}
        for sequence, probability in possible_sequences.items():
            possible_next = search_engine.get_next_token(sequence)
            if not possible_next:
                continue
            extended = search_engine.continue_sequence(
                sequence, possible_next, {sequence: probability}
            )
            if not extended:
                continue
            for extended_seq, extended_prob in extended.items():
                if (
                    extended_seq not in updated_sequences or 
                    extended_prob < updated_sequences[extended_seq]
                ):
                    updated_sequences[extended_seq] = extended_prob
        if not updated_sequences:
            break
        filtered = search_engine.prune_sequence_candidates(updated_sequences)
        possible_sequences = filtered or {}
        if not possible_sequences:
            break
    if possible_sequences:
        optimal_sequence = min(possible_sequences.items(), key=lambda x: x[1])[0]
        context_length = len(current_context)
        predicted_tokens = []
        for token_id in optimal_sequence[context_length:]:
            word = processor.get_token(token_id)
            if word and word != '<EOS>':
                predicted_tokens.append(word)
        if predicted_tokens:
            missing_section = " ".join(predicted_tokens)
            split_words = missing_section.split()
            if all(len(w) == 1 for w in split_words):
                missing_section = missing_section.replace(" ", "")
            completed_text = encrypted_text.replace("<BURNED>", missing_section)
            print(f"\nComplete letter:\n{completed_text}")

    results = (gb_result, bb_result, ga_result, ba_result, dynamic_result)
    assert all(results), "Result is None"

if __name__ == "__main__":
    main()
