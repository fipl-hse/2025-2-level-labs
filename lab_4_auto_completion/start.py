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

    print("\nSolution of the 3rd secret")
    n_gram_size_secret = 4
    beam_width = 6
    seq_len = 10
    with open("./assets/secrets/secret_3.txt", "r", encoding="utf-8") as file:
        secret_text = file.read()
    burned_pos = secret_text.find("<BURNED>")
    if burned_pos == -1:
        return
    lines = secret_text.split('\n')
    prompt_text = ""
    for line in lines:
        if "<BURNED>" in line:
            before_burned = line.split("<BURNED>")[0].strip()
            before_burned = before_burned.replace('"', '').replace('"', '').strip()
            words = before_burned.split()
            if words:
                prompt_text = before_burned.lower()
            break
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as f:
        hp_text = f.read()
    word_processor = WordProcessor(end_of_sentence_token="<EOS>")
    encoded_sentences = word_processor.encode_sentences(hp_text)
    flat_word_sequence = []
    for sentence in encoded_sentences:
        flat_word_sequence.extend(sentence)
    flat_word_sequence = tuple(flat_word_sequence)
    word_model = NGramLanguageModel(flat_word_sequence, n_gram_size_secret)
    if word_model.build() != 0:
        return
    encoded_prompt = word_processor.encode(prompt_text)
    if not encoded_prompt:
        return
    beam_searcher = BeamSearcher(beam_width, word_model)
    sequences = {encoded_prompt: 0.0}
    for step in range(seq_len):
        new_sequences = {}
        for seq, score in sequences.items():
            next_tokens = beam_searcher.get_next_token(seq)
            if not next_tokens:
                new_sequences[seq] = score
                continue
            for token_id, probability in next_tokens:
                if probability <= 0:
                    continue
                new_seq = seq + (token_id,)
                new_score = score - math.log(probability)
                new_sequences[new_seq] = new_score
        if not new_sequences:
            break
        sorted_sequences = sorted(new_sequences.items(), key=lambda x: x[1])
        sequences = dict(sorted_sequences[:beam_width])
    best_seq, _ = min(sequences.items(), key=lambda x: x[1])
    all_words = []
    for token_id in best_seq:
        word = word_processor.get_token(token_id)
        if word and word != '<EOS>':
            all_words.append(word)
    prompt_words = prompt_text.split()
    generated_words = all_words[len(prompt_words):]
    burned_part = " ".join(generated_words).strip()    
    burned_part = burned_part.replace("<EOS>", "").strip()
    burned_part = burned_part.rstrip('.!?,;')
    if burned_part and burned_part[0].isupper():
        burned_part = burned_part[0].lower() + burned_part[1:]
    print(f"\nGenerated text: '{burned_part}'")
    result = secret_text.replace("<BURNED>", burned_part)
    print("\nComplete letter:")
    print(result)

    results = (gb_result, bb_result, ga_result, ba_result, dynamic_result)
    assert all(results), "Result is None"

if __name__ == "__main__":
    main()
