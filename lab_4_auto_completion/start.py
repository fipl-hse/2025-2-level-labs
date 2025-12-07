"""
Auto-completion start
"""

# pylint:disable=unused-variable
from lab_3_generate_by_ngrams.main import (
    BeamSearcher,
    BeamSearchTextGenerator,
    GreedyTextGenerator,
    NGramLanguageModel
)
from lab_4_auto_completion.main import NGramTrieLanguageModel, WordProcessor, PrefixTrie


def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/hp_letters.txt", "r", encoding="utf-8") as letters_file:
        hp_letters = letters_file.read()
    with open("./assets/ussr_letters.txt", "r", encoding="utf-8") as text_file:
        ussr_letters = text_file.read()
    with open("./assets/secrets/secret_1.txt", "r", encoding="utf-8") as text_file:
        letter = text_file.read()
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as harry_file:
        text = harry_file.read()

    processor = WordProcessor('<EOS>')
    hp_encoded_sentences = processor.encode_sentences(hp_letters)
    prefix_trie = PrefixTrie()
    prefix_trie.fill(hp_encoded_sentences)
    suggestion = prefix_trie.suggest((1, 2))
    print(suggestion)
    test_text = "Hello world."
    encoded = processor.encode_sentences(test_text)
    print(f"Encoded: {encoded}")
    if encoded:
        decoded = processor.decode(encoded[0])
        print(f"Decoded: {decoded}")
    suggestions = prefix_trie.suggest((2,))
    if suggestions:
        first_suggestion = suggestions[0]
        decoded_string = processor.decode(first_suggestion)
        cleaned_result = decoded_string.replace("<EOS>", "").strip()
        print(cleaned_result)

    model = NGramTrieLanguageModel(hp_encoded_sentences, 5)
    model.build()

    greedy_res = GreedyTextGenerator(model, processor).run(55, 'Harry Potter')
    print(greedy_res)
    beam_res = BeamSearchTextGenerator(model, processor, 5).run('Harry Potter', 55)
    print(beam_res)

    encoded_ussr_sentences = processor.encode_sentences(ussr_letters)
    model.update(encoded_ussr_sentences)

    greedy_upd = GreedyTextGenerator(model, processor).run(55, 'Harry Potter')
    print(greedy_upd)
    beam_upd = BeamSearchTextGenerator(model, processor, 5).run('Harry Potter', 55)
    print(beam_upd)

    encoded_sentences = processor.encode_sentences(text)
    encoded_secret = []
    for sentence in encoded_sentences:
        encoded_secret.extend(sentence)
    encoded_secret = tuple(encoded_secret)
    print(encoded_secret)

    n_gram_size = 3
    beam_width = 7
    seq_len = 10
    test_ids = (306, 307, 0)  # hello, world, <EOS>
    decoded = processor.decode(test_ids)
    print(f"Direct decode test: {decoded}")
    language_model = NGramLanguageModel(encoded_secret, n_gram_size)
    language_model.build()

    letter_parts = letter.split("<BURNED>")
    first_part = letter_parts[0].strip()
    
    encoded_context = processor.encode_sentences(first_part)
    context = []
    for sentence in encoded_context:
        for token_id in sentence:
            token = processor.get_token(token_id)
            if token != '<EOS>':
                context.append(token_id)

    if len(context) >= (n_gram_size - 1):
        context = context[-(n_gram_size - 1):]
    context = tuple(context)
    print(f'context: {context}')

    beam_generator = BeamSearchTextGenerator(language_model, processor, beam_width)

    context_words = [processor.get_token(tid) for tid in context]
    context_str = " ".join([w for w in context_words if w])
    generated_text = beam_generator.run(context_str, seq_len)
    print(context_words, context_str)
    print(f'beamsearchgenerator: {generated_text}')
          
    beam_searcher = BeamSearcher(beam_width, language_model)
    sequence_candidates = {context: 0.0}
    for _ in range(seq_len):
        new_candidates = {}
        for sequence, probability in sequence_candidates.items():
            next_tokens = beam_searcher.get_next_token(sequence)
            if not next_tokens:
                continue
            updated_seq = beam_searcher.continue_sequence(sequence,
                                                          next_tokens, {sequence: probability})
            if not updated_seq:
                continue
            for new_seq, new_prob in updated_seq.items():
                if new_seq not in new_candidates or new_prob < new_candidates[new_seq]:
                    new_candidates[new_seq] = new_prob
        if not new_candidates:
            break
        pruned = beam_searcher.prune_sequence_candidates(new_candidates)
        if not pruned:
            break
        sequence_candidates = pruned or {}

    best_sequence = min(sequence_candidates.items(), key=lambda x: x[1])[0]
    generated_ids = best_sequence[len(context):]
    print(generated_ids)
    generated_words = []
    for token_id in generated_ids:
        token = processor.get_token(token_id)
        if token and token != "<EOS>":
            generated_words.append(token)
    if generated_words:
        burned = " ".join(generated_words)
    res_letter = letter.replace("<BURNED>", burned)
    print(res_letter)

    result = res_letter
    assert result, "Result is None"


if __name__ == "__main__":
    main()
