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
from lab_4_auto_completion.main import NGramTrieLanguageModel, WordProcessor


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

    word_processor = WordProcessor("<EOS>")
    # hp_encoded_corpus = word_processor.encode_sentences(hp_letters)
    # print(hp_encoded_corpus)
    sentences = word_processor.encode_sentences(text)
    if sentences:
        corpus = []
        for sentence in sentences:
            corpus.extend(sentence)
        hp_encoded_corpus = tuple(corpus)
    else:
        hp_encoded_corpus = tuple()

    n_gram_size = 3
    beam_width = 7
    seq_len = 10
    model = NGramLanguageModel(hp_encoded_corpus, n_gram_size)
    build_result = model.build()
    print(build_result)

    letter_parts = letter.split("<BURNED>")
    first_part = letter_parts[0].strip()
    encoded_context = word_processor.encode_sentences(first_part)
    context = []
    for sentence in encoded_context:
        context.extend(sentence)
    if len(context) >= (n_gram_size - 1):
        gen_context = tuple(context[-(n_gram_size - 1):])
    else:
        gen_context = tuple(context)

    beam_searcher = BeamSearcher(beam_width, model)

    sequence_candidates = {gen_context: 0.0}
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
    if len(best_sequence) > len(gen_context):
        generated_ids = best_sequence[len(gen_context):]
        generated_words = []
        for token_id in generated_ids:
            token = word_processor.get_token(token_id)
            if token and token != "<EOS>":
                generated_words.append(token)
        burned = " ".join(generated_words)
        res_letter = letter.replace("<BURNED>", burned)
        print(res_letter)
    # beam_generator = BeamSearchTextGenerator(model, word_processor, beam_width)
    # test_prompt = "harry potter"
    
    # beam_result = beam_generator.run(test_prompt, seq_len)
    # print(beam_result)
    result = hp_encoded_corpus
    assert result, "Result is None"


if __name__ == "__main__":
    main()
