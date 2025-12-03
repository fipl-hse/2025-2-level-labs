"""
Auto-completion start
"""

# pylint:disable=unused-variable
from lab_3_generate_by_ngrams.main import BeamSearcher, GreedyTextGenerator, NGramLanguageModel
from lab_4_auto_completion.main import WordProcessor


def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/hp_letters.txt", "r", encoding="utf-8") as letters_file:
        hp_letters = letters_file.read()
    with open("./assets/ussr_letters.txt", "r", encoding="utf-8") as text_file:
        ussr_letters = text_file.read()
    with open("./assets/secrets/secret_5.txt", "r", encoding="utf-8") as secret_file:
        secret = secret_file.read()
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as hp_file:
        text = hp_file.read()
    n_gram_size = 4
    beam_width = 3
    seq_len = 25

    processor = WordProcessor('<EoS>')
    encoded_sentences = processor.encode_sentences(text)
    encoded_secret = []
    for sentence in encoded_sentences:
        encoded_secret.extend(sentence)
    encoded_secret = tuple(encoded_secret)

    model = NGramLanguageModel(encoded_secret, n_gram_size)
    print(model.build())

    letter_parts = secret.split("<BURNED>")
    before_part = letter_parts[0].strip()

    encoded_context = processor.encode_sentences(before_part)
    context = []
    for sentence in encoded_context:
        for token_id in sentence:
            token = processor.get_token(token_id)
            if token != '<EoS>':
                context.append(token_id)
    context = tuple(context)

    algorithm = BeamSearcher(beam_width, model)
    sequence_candidates = {context: 0.0}

    for _ in range(seq_len):
        new_candidates = {}
        for sequence, probability in sequence_candidates.items():
            next_tokens = algorithm.get_next_token(sequence)
            if not next_tokens:
                continue
            updated = algorithm.continue_sequence(sequence, next_tokens, {sequence: probability})
            if not updated:
                continue
            for new_seq, new_prob in updated.items():
                if new_seq not in new_candidates or new_prob < new_candidates[new_seq]:
                    new_candidates[new_seq] = new_prob
        if not new_candidates:
            break
        pruned = algorithm.prune_sequence_candidates(new_candidates)
        sequence_candidates = pruned or {}
        if not sequence_candidates:
            break
    if sequence_candidates:
        best_seq = min(sequence_candidates.items(), key=lambda x: x[1])[0]
        context_len = len(context)
        new_tokens = []
        for token_id in best_seq[context_len:]:
            token = processor.get_token(token_id)
            if token and token != '<EoS>':
                new_tokens.append(token)
        if new_tokens:
            burned_part = " ".join(new_tokens)
            words = burned_part.split()
            if all(len(word) == 1 for word in words):
                burned_part = burned_part.replace(" ", "")
            completed_letter = secret.replace("<BURNED>", burned_part)
            print(f"\nThe whole letter: {completed_letter}")
    result = completed_letter
    assert result, "Result is None"

if __name__ == "__main__":
    main()
