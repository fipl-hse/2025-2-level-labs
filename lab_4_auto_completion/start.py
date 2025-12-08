"""
Auto-completion start
"""

# pylint:disable=unused-variable
from lab_3_generate_by_ngrams.main import (
    BackOffGenerator,
    BeamSearchTextGenerator,
    GreedyTextGenerator
)
from lab_4_auto_completion.main import (
    DynamicBackOffGenerator,
    DynamicNgramLMTrie,
    load,
    save,
    WordProcessor,
    NGramTrieLanguageModel
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


    processor = WordProcessor("<EoS>")
    encoded_corpus = processor.encode_sentences(hp_letters)

    language_model = DynamicNgramLMTrie(encoded_corpus, 5)
    language_model.build()

    path = r"C:\Users\artem\hse\2025-2-level-labs\lab_4_auto_completion\assets\dynamic_trie.json"

    save(language_model, path)
    loaded_model = load(path)


    generator = DynamicBackOffGenerator(loaded_model, processor)

    seq_len = 50
    prompt = "Ivanov"

    print(f"Dynamic BackOFF with 1 corpus: {generator.run(seq_len, prompt)}")

    encoded_corpus = processor.encode_sentences(ussr_letters)
    loaded_model.update(encoded_corpus)

    print(f"Dynamic BackOFF with 2 corpuses: {generator.run(seq_len, prompt)}")

    encoded_corpus = processor.encode_sentences(hp_letters + ussr_letters)
    language_model = NGramTrieLanguageModel(encoded_corpus, 5)
    language_model.build()

    # backoff = BackOffGenerator(language_model, processor)
    beamsearch = BeamSearchTextGenerator(language_model, processor, 5)
    greedy = GreedyTextGenerator(language_model, processor)

    # backoff_out = backoff.run(seq_len, prompt)
    beamsearch_out = beamsearch.run(prompt, seq_len)
    greedy_out = greedy.run(seq_len, prompt)

    # print(f"Backoff: {backoff_out}")
    print(f"BeamSearch: {beamsearch_out}")
    print(f"Greedy: {greedy_out}")

    result = loaded_model
    assert result


if __name__ == "__main__":
    main()
