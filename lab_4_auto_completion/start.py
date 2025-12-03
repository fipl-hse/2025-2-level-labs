"""
Auto-completion start
"""

# pylint:disable=unused-variable
from lab_3_generate_by_ngrams.main import (
    BeamSearcher,
    BeamSearchTextGenerator,
    GreedyTextGenerator,
    NGramLanguageModel,
)
from lab_4_auto_completion.main import (
    NGramTrieLanguageModel,
    PrefixTrie,
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
    with open("./assets/secrets/secret_2.txt", "r", encoding="utf=8") as secret_file:
        secret = secret_file.read()
    with open("./assets/Harry_Potter.txt", "r", encoding="utf=8") as harry_file:
        text = harry_file.read()

    processor = WordProcessor('<EOS>')
    encoded_sentences = processor.encode_sentences(hp_letters)

    prefix_trie = PrefixTrie()
    prefix_trie.fill(encoded_sentences)
    suggestions = prefix_trie.suggest((2,))
    if suggestions:
        first_suggestion = suggestions[0]
        decoded_string = processor.decode(first_suggestion)
        cleaned_result = decoded_string.replace("<EOS>", "").strip()
        print(cleaned_result)

    model = NGramTrieLanguageModel(encoded_sentences, 5)
    model.build()

    greedy_before = GreedyTextGenerator(model, processor)
    gb_result = greedy_before.run(52, 'Harry Potter')
    print(f"Greedy Generator befor: {gb_result}")

    beam_before = BeamSearchTextGenerator(model, processor, 3)
    bb_result = beam_before.run('Harry Potter', 52)
    print(f"Beam Generator befor: {bb_result}")

    encoded_ussr_sentences = processor.encode_sentences(ussr_letters)
    model.update(encoded_ussr_sentences)

    greedy_after = GreedyTextGenerator(model, processor)
    ga_result = greedy_after.run(52, 'Harry Potter')
    print(f"Greedy Generator after: {ga_result}")

    beam_after = BeamSearchTextGenerator(model, processor, 3)
    ba_result = beam_after.run('Harry Potter', 52)
    print(f"Beam Generator after: {ba_result}")

    result = (gb_result, bb_result, ga_result, ba_result)

    'DECODING SECRET 2'
    n_gram_size = 3
    beam_width = 5
    seq_len = 15

    encoded_sentences = processor.encode_sentences(text)
    encoded_secret = []
    for sentence in encoded_sentences:
        encoded_secret.extend(sentence)
    encoded_secret = tuple(encoded_secret)

    language_model = NGramLanguageModel(encoded_secret, n_gram_size)
    language_model.build()

    letter_parts = secret.split("<BURNED>")
    first_part = letter_parts[0].strip()

    encoded_context = processor.encode_sentences(first_part)
    context = []
    for sentence in encoded_context:
        for token_id in sentence:
            token = processor.get_token(token_id)
            if token != '<EOS>':
                context.append(token_id)
    context = tuple(context)

    beam_searcher = BeamSearcher(beam_width, language_model)
    sequence_candidates = {context: 0.0}

    for _ in range(seq_len):
        new_candidates = {}
        for seq, prob in sequence_candidates.items():
            next_tokens = beam_searcher.get_next_token(seq)
            if not next_tokens:
                continue

            updated = beam_searcher.continue_sequence(seq, next_tokens, {seq: prob})
            if not updated:
                continue

            for new_seq, new_prob in updated.items():
                if new_seq not in new_candidates or new_prob < new_candidates[new_seq]:
                    new_candidates[new_seq] = new_prob

        if not new_candidates:
            break

        pruned = beam_searcher.prune_sequence_candidates(new_candidates)
        sequence_candidates = pruned or {}
        if not sequence_candidates:
            break

    if sequence_candidates:
        best_seq = min(sequence_candidates.items(), key=lambda x: x[1])[0]
        context_len = len(context)
        new_tokens = []

        for token_id in best_seq[context_len:]:
            token = processor.get_token(token_id)
            if token and token != '<EOS>':
                new_tokens.append(token)

        if new_tokens:
            burned_part = " ".join(new_tokens)
            words = burned_part.split()
            if all(len(w) == 1 for w in words):
                burned_part = burned_part.replace(" ", "")
            restored = secret.replace("<BURNED>", burned_part)
            print(f"\nRESTORED SECRET 2:\n{restored}")
    assert result, "Result is None"


if __name__ == "__main__":
    main()
