"""
Auto-completion start
"""

# pylint:disable=unused-variable
from lab_3_generate_by_ngrams.main import (
    BeamSearcher,
    BeamSearchTextGenerator,
    GreedyTextGenerator,
    NGramLanguageModel,
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
    with open("./assets/secrets/secret_1.txt", "r", encoding="utf-8") as text_file:
        letter = text_file.read()
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as harry_file:
        text = harry_file.read()

    processor = WordProcessor('<EOS>')
    hp_encoded_sentences = processor.encode_sentences(hp_letters)
    prefix_trie = PrefixTrie()
    prefix_trie.fill(hp_encoded_sentences)
    suggestions = prefix_trie.suggest((1,))
    if suggestions:
        first_suggestion = suggestions[0]
        decoded_string = processor.decode(first_suggestion)
        cleaned_result = decoded_string.replace("<EOS>", "").strip()
        print(cleaned_result)

    model = NGramTrieLanguageModel(hp_encoded_sentences, 5)
    model.build()

    greedy_res = GreedyTextGenerator(model, processor).run(55, 'Harry Potter')
    print(greedy_res)
    beam_res = BeamSearchTextGenerator(model, processor, 5).run('Harry Potter', 55)
    print(beam_res)

    encoded_ussr_sentences = processor.encode_sentences(ussr_letters)
    model.update(encoded_ussr_sentences)

    greedy_upd = GreedyTextGenerator(model, processor).run(55, 'Harry Potter')
    print(greedy_upd)
    beam_upd = BeamSearchTextGenerator(model, processor, 5).run('Harry Potter', 55)
    print(beam_upd)

    result = [greedy_res, beam_res, greedy_upd, beam_upd]
    assert result, "Result is None"


if __name__ == "__main__":
    main()
