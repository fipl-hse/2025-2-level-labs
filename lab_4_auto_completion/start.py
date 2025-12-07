"""
Auto-completion start
"""

# pylint:disable=unused-variable
from lab_3_generate_by_ngrams.main import BeamSearchTextGenerator, GreedyTextGenerator
from lab_4_auto_completion.main import (
    DynamicBackOffGenerator,
    DynamicNgramLMTrie,
    load,
    NGramTrieLanguageModel,
    PrefixTrie,
    save,
    WordProcessor,
)


def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/hp_letters.txt", "r", encoding="utf-8") as f:
        hp_letters = f.read()
    with open("./assets/ussr_letters.txt", "r", encoding="utf-8") as f:
        ussr_letters = f.read()
    #with open("./assets/secrets/secret_5.txt", "r", encoding="utf-8") as text_file:
        #secret = text_file.read()
    #with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as harry_file:
        #text = harry_file.read()
    processor = WordProcessor('<EOS>')
    hp_encoded = processor.encode_sentences(hp_letters)
    trie = PrefixTrie()
    trie.fill(hp_encoded)
    suggestion = trie.suggest((2,))
    if suggestion:
        storage = getattr(processor, '_storage', {})
        decoded = [word for token_id in suggestion[0] 
                  for word, word_id in storage.items() 
                  if word_id == token_id]
        print(f"\n1. Decoded result: {' '.join(decoded)}")
    model = NGramTrieLanguageModel(hp_encoded, 5)
    model.build()
    print(f"\n2. Greedy result before: {GreedyTextGenerator(model, processor).run(52, 'Dear')}")
    print(f"Beam result before: {BeamSearchTextGenerator(model, processor, 3).run('Dear', 52)}")
    encoded_ussr = processor.encode_sentences(ussr_letters)
    model.update(encoded_ussr)
    print(f"\n3. Greedy result after: {GreedyTextGenerator(model, processor).run(52, 'Dear')}")
    print(f"Beam result after: {BeamSearchTextGenerator(model, processor, 3).run('Dear', 52)}")
    dynamic_trie = DynamicNgramLMTrie(hp_encoded, 5)
    dynamic_trie.build()
    save(dynamic_trie, "./saved_dynamic_trie.json")
    loaded_trie = load("./saved_dynamic_trie.json")
    dynamic_generator = DynamicBackOffGenerator(loaded_trie, processor)
    print(f"\n4. Dynamic result before: {dynamic_generator.run(50, 'Ivanov')}")
    loaded_trie.update(encoded_ussr)
    print(f"Dynamic result after: {dynamic_generator.run(50, 'Ivanov')}")
    result = dynamic_generator.run(15, 'Dear')
    assert result, "Result is None"


if __name__ == "__main__":
    main()
