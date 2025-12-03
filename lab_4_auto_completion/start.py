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

    language_model = NGramLanguageModel(encoded_secret, n_gram_size)
    print(language_model.build())

    letter_parts = secret.split("<BURNED>")
    first_part = letter_parts[0].strip()

    encoded_context = processor.encode_sentences(first_part)
    context = []
    for sentence in encoded_context:
        for token_id in sentence:
            token = processor.get_token(token_id)
            if token != '<EoS>':
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
            if token and token != '<EoS>':
                new_tokens.append(token)
        if new_tokens:
            burned_part = " ".join(new_tokens)
            words = burned_part.split()
            if all(len(word) == 1 for word in words):
                burned_part = burned_part.replace(" ", "")
            restored = secret.replace("<BURNED>", burned_part)
            print(f"\nThe whole letter: {restored}")

    result = restored
    assert result, "Result is None"

if __name__ == "__main__":
    main()
