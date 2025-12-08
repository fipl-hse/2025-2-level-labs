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
    decoded_corpus = []
    if suggestions:
        main_suggestion = suggestions[0]
        for word in main_suggestion:
            decoded_corpus.append(word_processor.get_token(word))
        sentences = " ".join(decoded_corpus).split(word_processor._end_of_sentence_token)
        result_sentences = [sentence.strip().capitalize() for sentence in sentences if sentence]
        text = ". ".join(result_sentences)
        print(text + ".")
    else:
        print("No suggestions")
    ngram_model = NGramTrieLanguageModel(encoded_hp_letters, 5)
    greedy_generator = GreedyTextGenerator(ngram_model, word_processor)
    beam_searcher = BeamSearchTextGenerator(ngram_model, word_processor, 3)

    print(f"\n Before merge: Greedy Text:{greedy_generator.run(40, "Dear")}")
    print(f"\n Before merge: Beam Search Text:{beam_searcher.run("Dear", 40)}")

    encoded_ussr_letters = word_processor.encode_sentences(ussr_letters)
    ngram_model.update(encoded_ussr_letters)

    print(f"\n Before merge: Greedy Text:{greedy_generator.run(40, "Dear")}")
    print(f"\n Before merge: Beam Search Text:{beam_searcher.run("Dear", 40)}")
    result = main_suggestion
    assert result, "Result is None"


if __name__ == "__main__":
    main()
