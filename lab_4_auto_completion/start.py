"""
Auto-completion start
"""

# pylint:disable=unused-variable
from lab_3_generate_by_ngrams.main import BeamSearchTextGenerator, GreedyTextGenerator
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

    processor = WordProcessor('<EOS>')
    hp_encoded_sentences = processor.encode_sentences(hp_letters)
    prefix_trie = PrefixTrie()
    prefix_trie.fill(hp_encoded_sentences)
    suggestions = prefix_trie.suggest((2,))
    if suggestions:
        decoded = processor.decode(suggestions[0])
        print(f"suggestion: {decoded.replace('<EOS>', '').strip()}")

    model = NGramTrieLanguageModel(hp_encoded_sentences, 5)
    model.build()

    greedy_res = GreedyTextGenerator(model, processor).run(50, 'Harry')
    beam_res = BeamSearchTextGenerator(model, processor, 3).run('Harry', 50)

    model.update(processor.encode_sentences(ussr_letters))
    greedy_upd_res = GreedyTextGenerator(model, processor).run(50, 'Harry')
    beam_upd_res = BeamSearchTextGenerator(model, processor, 3).run('Harry', 50)
    print(greedy_res)
    print(greedy_upd_res)
    print(beam_res)
    print(beam_upd_res)
    result = [greedy_res, beam_res, greedy_upd_res, beam_upd_res]
    assert result, "Result is None"


if __name__ == "__main__":
    main()
