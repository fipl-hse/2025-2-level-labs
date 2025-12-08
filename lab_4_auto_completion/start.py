"""
Auto-completion start
"""
from lab_3_generate_by_ngrams.main import BeamSearchTextGenerator, GreedyTextGenerator
from lab_4_auto_completion.main import NGramTrieLanguageModel, PrefixTrie, WordProcessor

# pylint:disable=unused-variable


def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/hp_letters.txt", "r", encoding="utf-8") as letters_file:
        hp_letters = letters_file.read()
    with open("./assets/ussr_letters.txt", "r", encoding="utf-8") as text_file:
        ussr_letters = text_file.read()

    word_processor = WordProcessor(".")
    encoded_hp_letters = word_processor.encode_sentences(hp_letters)

    prefix_trie = PrefixTrie()
    prefix_trie.fill(encoded_hp_letters)
    suggestions = prefix_trie.suggest((2,))
    if suggestions:
        decoded_corpus = []
        for word in suggestions[0]:
            if word_processor.get_token(word) is not None:
                decoded_corpus.append(word_processor.get_token(word))
        sentences = " ".join(decoded_corpus).split(".")
        result_sentences = [sentence.strip().capitalize() for sentence in sentences if sentence]
        print(". ".join(result_sentences) + ".")
    else:
        print("No suggestions")
    model = NGramTrieLanguageModel(encoded_hp_letters, 5)
    model.build()

    print(f"\n Before: Greedy:{GreedyTextGenerator(model, word_processor).run(40, "Dear")}")
    print(
        f"\n Before: BeamSearch:{BeamSearchTextGenerator(model, word_processor, 3).run("Dear", 40)}"
        )

    model.update(word_processor.encode_sentences(ussr_letters))

    print(f"\n After: Greedy:{GreedyTextGenerator(model, word_processor).run(40, "Dear")}")
    print(
        f"\n After: BeamSearch:{BeamSearchTextGenerator(model, word_processor, 3).run("Dear", 40)}"
        )
    result = suggestions[0]
    assert result, "Result is None"


if __name__ == "__main__":
    main()
