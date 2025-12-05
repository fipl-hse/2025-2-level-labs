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

if __name__ == "__main__":
    main()
