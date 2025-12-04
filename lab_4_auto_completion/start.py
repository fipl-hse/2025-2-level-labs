"""
Auto-completion start
"""

# pylint:disable=unused-variable

from lab_3_generate_by_ngrams.main import NGramLanguageModel, BeamSearcher
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
    word_processor = WordProcessor(end_of_sentence_token="<EOS>")
    encoded_hp = word_processor.encode_sentences(hp_letters)
    prefix_trie = PrefixTrie()
    decoded_sentence = ""
    if encoded_hp:
        prefix_trie.fill(encoded_hp)
        suggestions = prefix_trie.suggest((2,))
        if suggestions:
            first_suggestion = suggestions[0]
            decoded_words = []
            reverse_storage = {v: k for k, v in word_processor._storage.items()}
            for token_id in first_suggestion:
                if token_id in reverse_storage:
                    decoded_words.append(reverse_storage[token_id])
            decoded_sentence = " ".join(decoded_words)
    n_gram_size = 5
    model = NGramTrieLanguageModel(encoded_hp, n_gram_size)
    build_result = model.build()
    if build_result == 0:
        prompt = "Dear Harry"
        encoded_ussr = word_processor.encode_sentences(ussr_letters)
        if encoded_ussr:
            model.update(encoded_ussr)
    dynamic_model = DynamicNgramLMTrie(encoded_hp, n_gram_size=5)
    dynamic_build_result = dynamic_model.build()
    if dynamic_build_result == 0:
        save(dynamic_model, "./dynamic_model.json")
        loaded_model = load("./dynamic_model.json")
        dynamic_generator = DynamicBackOffGenerator(loaded_model, word_processor)
        dynamic_result = dynamic_generator.run(50, "Ivanov")
        if encoded_ussr:
            loaded_model.update(encoded_ussr)
            dynamic_result_after = dynamic_generator.run(50, "Ivanov")
    result = decoded_sentence
    assert result, "Result is None"

    print("\nSolution of the 3rd secret")
    n_gram_size_secret = 4
    beam_width = 6
    seq_len = 10
    secret_path = "./assets/secrets/secret_3.txt"
    try:
        with open(secret_path, "r", encoding="utf-8") as file:
            secret_text = file.read()
    except FileNotFoundError:
        return
    print("\nOriginal secret text:")
    print(secret_text)
    burned_start = secret_text.find("<BURNED>")
    if burned_start == -1:
        return
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
    print(f"\nGenerated text: '{burned_part}'")
    restored_text = secret_text.replace("<BURNED>", burned_part)
    output_path = "restored_secret_3.txt"
    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write(restored_text)
    print("\nComplete letter:")
    print(restored_text)

if __name__ == "__main__":
    main()
