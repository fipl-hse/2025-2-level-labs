"""
Auto-completion start
"""

from lab_3_generate_by_ngrams.main import BeamSearchTextGenerator, GreedyTextGenerator

# pylint:disable=unused-variable
from lab_4_auto_completion.main import NGramTrieLanguageModel, PrefixTrie, WordProcessor


def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/hp_letters.txt", "r", encoding="utf-8") as letters_file:
        hp_letters = letters_file.read()
    with open("./assets/ussr_letters.txt", "r", encoding="utf-8") as text_file:
        ussr_letters = text_file.read()

    word_processor = WordProcessor("<EoS>")
    hp_encoded_text = word_processor.encode_sentences(hp_letters)
    prefix_trie = PrefixTrie()
    prefix_trie.fill(hp_encoded_text)
    suggestions = prefix_trie.suggest((2,))
    if suggestions:
        print("For 6: ", word_processor.decode(suggestions[0]).replace("<EoS>", "").strip())


    ngram_trie_lm = NGramTrieLanguageModel(hp_encoded_text, 5)
    ngram_trie_lm.build()

    ussr_encoded_text = word_processor.encode_sentences(ussr_letters)

    greedy_generator = GreedyTextGenerator(ngram_trie_lm, word_processor)
    beamsearcher = BeamSearchTextGenerator(ngram_trie_lm, word_processor, 3)

    ussr_corp_before = (
        greedy_generator.run(seq_len=30, prompt="Dear"),
        beamsearcher.run(prompt="Dear", seq_len=30)
        )
    print(f"Greedy before update = {ussr_corp_before[0]}, Beam = {ussr_corp_before[1]}")
    print()
    # print("Beamsearcher before update", ussr_corp_beam)
    print()

    ngram_trie_lm.update(ussr_encoded_text)
    ussr_corp_after = (
        greedy_generator.run(seq_len=30, prompt="Dear"),
        beamsearcher.run(prompt="Dear", seq_len=30)
        )
    print(f"Greedy after update = {ussr_corp_before[0]}, Beam = {ussr_corp_before[1]}")
    print()


    result = ussr_corp_after
    print(result)
    assert result, "Result is None"


if __name__ == "__main__":
    main()
