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
    result = None


    processor = WordProcessor('<EOS>')
    trie = PrefixTrie()
    trie.fill(processor.encode_sentences(hp_letters))
    suggestions = trie.suggest((2,))
    if suggestions:
        processor.decode(suggestions[0])
        print(f"First sequence for prefix 2: {processor.decode(suggestions[0])}")

    model = NGramTrieLanguageModel(processor.encode_sentences(hp_letters), 5)
    model.build()

    greedy_generator = GreedyTextGenerator(model, processor)
    beam_generator = BeamSearchTextGenerator(model, processor, 5)

    greedy_before = greedy_generator.run(15, "Harry Potter")
    print(f"Greedy text generator before merging: {greedy_before}")


    beam_before = beam_generator.run("Harry Potter", 15)
    print(f"Beam search text generator before merging: {beam_before}")


    print("Merging corpuses...")
    model.update(processor.encode_sentences(ussr_letters))


    greedy_after = greedy_generator.run(15, "Harry Potter")
    print(f"Greedy text generator after merging: {greedy_after}")

    beam_after = beam_generator.run("Harry Potter", 15)
    print(f"Beam search text generator after merging: {beam_after}")

    result = beam_after

    print(result)

    assert result, "Result is None"
if __name__ == "__main__":
    main()
