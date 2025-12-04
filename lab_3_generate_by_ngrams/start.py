"""
Generation by NGrams starter
"""

# pylint:disable=unused-import, unused-variable
from lab_3_generate_by_ngrams.main import (
    BackOffGenerator,
    BeamSearcher,
    BeamSearchTextGenerator,
    GreedyTextGenerator,
    NGramLanguageModel,
    NGramLanguageModelReader,
    TextProcessor,
)

from lab_4_auto_completion.main import WordProcessor


def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    processor = TextProcessor(end_of_word_token='_')
    encoded = processor.encode(text) or tuple()
    if encoded:
        loaded_model_list = []
        for n_size in range(2, 11):
            custom_model = NGramLanguageModel(encoded, n_size)
            custom_model.build()
            loaded_model_list.append(custom_model)
            result = GreedyTextGenerator(
                custom_model,
                processor
            ).run(30, 'Harry')
            print(f"\nGreedy with custom {n_size}-gram: {result}")
            result = BeamSearchTextGenerator(
                custom_model,
                processor,
                2
            ).run('Harry', 30)
            print(f"\nBeam Search with custom {n_size}-gram: {result}")
        if loaded_model_list:
            result = BackOffGenerator(
                tuple(loaded_model_list),
                processor
            ).run(35, 'Harry')
            print(f"\nBackOff with all custom models: {result}")
            assert result

    print("\nSolution of the 3rd secret")
    n_gram_size_secret = 4
    beam_width = 6
    seq_len = 10
    secret_path = "../lab_4_auto_completion/assets/secrets/secret_3.txt"
    with open(secret_path, "r", encoding="utf-8") as file:
        secret_text = file.read()
    print("\nOriginal secret text:")
    print(secret_text)
    burned_start = secret_text.find("<BURNED>")
    word_processor = WordProcessor(end_of_sentence_token="<EOS>")
    context_start = max(0, burned_start - 500)
    context_text = secret_text[context_start:burned_start]
    encoded_context = word_processor.encode_sentences(context_text)
    context_ids = []
    for sentence in encoded_context:
        for token_id in sentence:
            token = word_processor.get_token(token_id)
            if token and token != '<EOS>':
                context_ids.append(token_id)
    if len(context_ids) >= n_gram_size_secret - 1:
        context = tuple(context_ids[-(n_gram_size_secret - 1):])
    else:
        context = tuple(context_ids)
    decoded_context_words = []
    reverse_storage = {v: k for k, v in word_processor._storage.items()}
    for token_id in context:
        if token_id in reverse_storage:
            decoded_context_words.append(reverse_storage[token_id])
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as hp_file:
        harry_text = hp_file.read()
    encoded_harry = word_processor.encode_sentences(harry_text)
    harry_corpus = []
    for sentence in encoded_harry:
        harry_corpus.extend(sentence)
    harry_corpus = tuple(harry_corpus)
    language_model = NGramLanguageModel(harry_corpus, n_gram_size_secret)
    build_result = language_model.build()
    if build_result != 0:
        return
    beam_searcher = BeamSearcher(beam_width, language_model)
    sequence_candidates = {context: 0.0}
    for step in range(seq_len):
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
    if not sequence_candidates:
        return
    best_seq = min(sequence_candidates.items(), key=lambda x: x[1])[0]
    context_len = len(context)
    new_tokens = []
    for token_id in best_seq[context_len:]:
        token = word_processor.get_token(token_id)
        if token and token != '<EOS>':
            new_tokens.append(token)
    if not new_tokens:
        return
    burned_part = " ".join(new_tokens).strip()
    if burned_part.endswith('.'):
        burned_part = burned_part[:-1]
    print(f"\nGenerated text: '{burned_part}'")
    restored_text = secret_text.replace("<BURNED>", burned_part)
    output_path = "restored_secret_3.txt"
    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write(restored_text)
    print("\nComplete letter:")
    print(restored_text)


if __name__ == "__main__":
    main()
