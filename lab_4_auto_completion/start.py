"""
Auto-completion start
"""

# pylint:disable=unused-variable
from lab_3_generate_by_ngrams.main import (
    BeamSearchTextGenerator,
    GreedyTextGenerator,
)
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
    with open("./assets/secrets/secret_4.txt", "r", encoding="utf=8") as secret_file:
        secret = secret_file.read()
    with open("./assets/Harry_Potter.txt", "r", encoding="utf=8") as harry_file:
        text = harry_file.read()

    processor = WordProcessor('<EOS>')
    encoded_sentences = processor.encode_sentences(hp_letters)

    prefix_trie = PrefixTrie()
    prefix_trie.fill(encoded_sentences)
    suggestions = prefix_trie.suggest((2,))
    if suggestions:
        first_suggestion = suggestions[0]
        id_to_word = {v: k for k, v in processor._storage.items()}
        words = []
        for word_id in first_suggestion:
            if word_id in id_to_word:
                word = id_to_word[word_id]
                if word != '<EOS>':
                    words.append(word)
        words_tuple = tuple(words)
        print(processor._postprocess_decoded_text(words_tuple))
    
    model = NGramTrieLanguageModel(encoded_sentences, 5)
    model.build()

    greedy_before = GreedyTextGenerator(model, processor)
    gb_result = greedy_before.run(52, 'Harry Potter')
    print(f"Greedy Generator befor: {gb_result}")

    beam_before = BeamSearchTextGenerator(model, processor, 3)
    bb_result = beam_before.run('Harry Potter', 52)
    print(f"Beam Generator befor: {bb_result}")

    encoded_ussr_sentences = processor.encode_sentences(ussr_letters)
    model.update(encoded_ussr_sentences)

    greedy_after = GreedyTextGenerator(model, processor)
    ga_result = greedy_after.run(52, 'Harry Potter')
    print(f"Greedy Generator after: {ga_result}")

    beam_after = BeamSearchTextGenerator(model, processor, 3)
    ba_result = beam_after.run('Harry Potter', 52)
    print(f"Beam Generator after: {ba_result}")
    
    result = (gb_result, bb_result, ga_result, ba_result)
    assert result, "Result is None"


if __name__ == "__main__":
    main()
