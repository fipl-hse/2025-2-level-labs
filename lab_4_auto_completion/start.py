"""
Auto-completion start
"""

# pylint:disable=unused-variable
from lab_3_generate_by_ngrams.main import BeamSearcher, BeamSearchTextGenerator, GreedyTextGenerator, NGramLanguageModel
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
    with open("./assets/secrets/secret_5.txt", "r", encoding="utf-8") as text_file:
        letter = text_file.read()
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    n_gram_size = 4
    beam_width = 3
    seq_len = 25

    processor = WordProcessor('.')
    encoded_sentences = processor.encode_sentences(text)

    encoded_corpus = []
    for sentence in encoded_sentences:
        encoded_corpus.extend(sentence)
    encoded_corpus = tuple(encoded_corpus)

    model = NGramLanguageModel(encoded_corpus, n_gram_size)
    print(model.build())

    algorithm = BeamSearcher(beam_width, model)

    parts = letter.split("<BURNED>")
    part_of_letter = parts[0]
    after_burned = parts[1]

    encoded_context_sentences = processor.encode_sentences(part_of_letter)

    context_sequence = []
    for sentence in encoded_context_sentences:
        context_sequence.extend(sentence)
    context_sequence = tuple(context_sequence)

    if len(context_sequence) >= n_gram_size - 1:
        initial_sequence = context_sequence[-(n_gram_size - 1):]
    else:
        initial_sequence = context_sequence

    sequence_candidates = {initial_sequence: 0.0}
    for _ in range(seq_len):
        new_candidates = {}
        for sequence, probability in sequence_candidates.items():
            next_tokens = algorithm.get_next_token(sequence)
            if next_tokens is None:
                return None
            updated_candidates = algorithm.continue_sequence(
                sequence, next_tokens, {sequence: probability})
            if updated_candidates is not None:
                for seq, prob in updated_candidates.items():
                    if seq not in new_candidates or prob < new_candidates[seq]:
                        new_candidates[seq] = prob
        if not new_candidates:
            break
        sequence_candidates = algorithm.prune_sequence_candidates(new_candidates) or {}
        if sequence_candidates is None:
            return None
    if not sequence_candidates:
        return None
    best_sequence = min(sequence_candidates.items(), key=lambda x: x[1])[0][(n_gram_size - 1):]
    decoded_words = [processor.get_token(token_id) for token_id in best_sequence if processor.get_token(token_id) is not None]
    if decoded_words:
        generated_text = processor._postprocess_decoded_text(tuple(decoded_words))
    else:
        generated_text = ""
    generated_text = processor._postprocess_decoded_text(tuple(decoded_words))
    completed_letter = part_of_letter + generated_text + after_burned
    print(f'\nGenerated text: {generated_text}')
    print(f'\nCompleted letter:\n{completed_letter}')
    result = completed_letter
    assert result, "Result is None"

if __name__ == "__main__":
    main()
