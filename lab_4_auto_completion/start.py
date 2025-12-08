"""
Auto-completion start
"""

from lab_3_generate_by_ngrams.main import BeamSearchTextGenerator, GreedyTextGenerator

# pylint:disable=unused-variable
from lab_4_auto_completion.main import (
    NGramTrieLanguageModel,
    PrefixTrie,
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
    result = None
    word_processor = WordProcessor(end_of_sentence_token="<EOS>")
    encoded_hp = word_processor.encode_sentences(hp_letters)
    prefix_trie = PrefixTrie()
    prefix_trie.fill(encoded_hp)
    suggestions = prefix_trie.suggest((2,))
    if suggestions:
        decoded = word_processor.decode(suggestions[0])
        print(decoded.replace("<EOS>", "").strip())
    ngram_model = NGramTrieLanguageModel(encoded_hp, 5)
    ngram_model.build()
    greedy_generator = GreedyTextGenerator(ngram_model, word_processor)
    beam_generator = BeamSearchTextGenerator(ngram_model, word_processor, beam_width=3)
    before = (greedy_generator.run(seq_len=30, prompt="Dear"),
              beam_generator.run(prompt="Dear", seq_len=30))
    ngram_model.update(word_processor.encode_sentences(ussr_letters))
    after = (greedy_generator.run(seq_len=30, prompt="Dear"),
             beam_generator.run(prompt="Dear", seq_len=30))
    print(f"Before update: Greedy = {before[0]}, Beam = {before[1]}")
    print(f"After update: Greedy = {after[0]}, Beam = {after[1]}")
    result = decoded
    assert result, "Result is None"


if __name__ == "__main__":
    main()
