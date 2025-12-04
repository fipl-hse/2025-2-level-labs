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
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()

    processor = WordProcessor('<EOS>')
    hp_encoded_sentences = processor.encode_sentences(hp_letters)
    prefix_trie = PrefixTrie()
    prefix_trie.fill(hp_encoded_sentences)

    encoded_sentences = processor.encode_sentences(text)
    encoded_secret = []
    for sentence in encoded_sentences:
        encoded_secret.extend(sentence)
    encoded_secret = tuple(encoded_secret)

    n_gram_size = 3
    beam_width = 7
    seq_len = 10
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
    context = tuple(context)

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
        sequence_candidates = pruned

    best_sequence = min(sequence_candidates.items(), key=lambda x: x[1])[0]
    if len(best_sequence) > len(context):
        generated_ids = best_sequence[len(context):]
        generated_words = []
        for token_id in generated_ids:
            token = processor.get_token(token_id)
            if token and token != "<EOS>":
                generated_words.append(token)
        burned = " ".join(generated_words)
        res_letter = letter.replace("<BURNED>", burned)
        print(res_letter)

    result = res_letter
    assert result, "Result is None"


if __name__ == "__main__":
    main()
