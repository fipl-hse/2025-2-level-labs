"""
Auto-completion start
"""

# pylint:disable=unused-variable
from lab_3_generate_by_ngrams.main import BackOffGenerator, BeamSearchTextGenerator, GreedyTextGenerator
from lab_4_auto_completion.main import (
    DynamicBackOffGenerator,
    DynamicNgramLMTrie,
    load,
    save,
    WordProcessor,
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

    lm = DynamicNgramLMTrie(encoded_corpus, 5)
    lm.build()

    path = r"C:\Users\harperentity\Desktop\University\2nd_year\Module_1\Programming_for_linguists\Fork\2025-2-level-labs\lab_4_auto_completion\assets\dynamic_trie.json"

    save(lm, path)
    loaded_model = load(path)

    generator = DynamicBackOffGenerator(loaded_model, processor)

    seq_len = 50
    prompt = "Ivanov"

    print(f" \nBefore merging corpuses: {generator.run(seq_len, prompt)}")

    print("Merging corpuses...")
    encoded_corpus = processor.encode_sentences(ussr_letters)
    loaded_model.update(encoded_corpus)

    print(f"After merging corpuses: {generator.run(seq_len, prompt)}")

    result = loaded_model
    assert result

if __name__ == "__main__":
    main()
