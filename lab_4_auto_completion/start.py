"""
Auto-completion start
"""

# pylint:disable=unused-variable

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

    print("\nSolution of the 2nd secret")
    with open("./assets/secrets/secret_2.txt", "r", encoding="utf-8") as secret_file:
        secret_content = secret_file.read()
    print("\nOriginal secret text:")
    print(secret_content)
    burned_start = secret_content.find("<BURNED>")
    if burned_start == -1:
        return
    before_burned = secret_content[:burned_start]
    words = before_burned.split()
    if len(words) >= 2:
        context_words = words[-2:]
        prompt_text = " ".join(context_words)
    else:
        prompt_text = ""
    print(f"\nContext for generation: '{prompt_text}'")
    prompt_words = prompt_text.lower().split()
    prompt_ids = []
    for word in prompt_words:
        if word in word_processor._storage:
            prompt_ids.append(word_processor._storage[word])
        else:
            clean_word = ''.join(c for c in word if c.isalpha())
            if clean_word in word_processor._storage:
                prompt_ids.append(word_processor._storage[clean_word])
    if not prompt_ids:
        return
    n_gram_size = 3
    ngrams = {}
    for sentence in encoded_hp:
        for i in range(len(sentence) - n_gram_size + 1):
            ngram = tuple(sentence[i:i + n_gram_size])
            ngrams[ngram] = ngrams.get(ngram, 0) + 1
    beam_width = 5
    seq_len = 15
    current_sequences = [(tuple(prompt_ids), 0.0)]
    for step in range(seq_len):
        new_sequences = []
        for seq, score in current_sequences:
            if len(seq) < n_gram_size - 1:
                continue
            context = seq[-(n_gram_size - 1):]
            candidates = {}
            for ngram, count in ngrams.items():
                if ngram[:n_gram_size - 1] == context:
                    next_word = ngram[-1]
                    candidates[next_word] = candidates.get(next_word, 0) + count
            if candidates:
                sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
                top_candidates = sorted_candidates[:beam_width]
                for token_id, freq in top_candidates:
                    prob = freq / sum(candidates.values())
                    new_seq = seq + (token_id,)
                    new_score = score - (1.0 - prob)
                    new_sequences.append((new_seq, new_score))
            else:
                new_sequences.append((seq, score))
        if not new_sequences:
            break
        new_sequences.sort(key=lambda x: x[1])
        current_sequences = new_sequences[:beam_width]
    if not current_sequences:
        return
    best_sequence, best_score = current_sequences[0]
    generated_ids = best_sequence[len(prompt_ids):]
    reverse_storage = {v: k for k, v in word_processor._storage.items()}
    generated_words = []
    for word_id in generated_ids:
        if word_id in reverse_storage:
            word = reverse_storage[word_id]
            if word != "<EOS>":
                generated_words.append(word)
    generated_text = " ".join(generated_words)
    if generated_text:
        generated_text = generated_text[0].upper() + generated_text[1:]
        if not generated_text.endswith(('.', '!', '?')):
            generated_text = generated_text.rstrip('.!?') + "."
        generated_text = generated_text.strip()
    else:
        generated_text = "no reaction at all."
    print(f"\nGenerated text: '{generated_text}'")
    completed_text = secret_content.replace("<BURNED>", generated_text)
    with open("./completed_secret_2.txt", "w", encoding="utf-8") as output_file:
        output_file.write(completed_text)
    print("\nComplete letter (with generated text):")
    print(completed_text)
    return completed_text

if __name__ == "__main__":
    main()
