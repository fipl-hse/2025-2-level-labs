"""
Auto-completion start
"""

# pylint:disable=unused-variable
from lab_3_generate_by_ngrams.main import (
    BeamSearchTextGenerator,
    GreedyTextGenerator
)
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
        first_suggestion = suggestions[0]
        decoded = processor.decode(first_suggestion)
        print(f"Prefix Trie suggestion: {decoded.replace('<EOS>', '').strip()}")

    model = NGramTrieLanguageModel(hp_encoded_sentences, 5)
    model.build()

    greedy_generator = GreedyTextGenerator(model, processor)
    greedy_res = greedy_generator.run(20, 'Dear Harry')
    beam_generator = BeamSearchTextGenerator(model, processor, 3)
    beam_res = beam_generator.run('Dear Harry', 20)
    
    ussr_encoded_sentences = processor.encode_sentences(ussr_letters)
    model.update(ussr_encoded_sentences)

    greedy_upd = GreedyTextGenerator(model, processor)
    greedy_upd_res = greedy_upd.run(20, 'Dear Harry')
    beam_upd = BeamSearchTextGenerator(model, processor, 3)
    beam_upd_res = beam_upd.run('Dear Harry', 20)
    
    print(greedy_res)
    print(greedy_upd_res)
    print(beam_res)
    print(beam_upd_res)

    result = [greedy_res, beam_res, greedy_upd, beam_upd]
    assert result, "Result is None"


if __name__ == "__main__":
    main()
