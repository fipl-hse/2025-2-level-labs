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
    encoded_sentences = processor.encode_sentences(hp_letters)
    prefix_trie = PrefixTrie()
    prefix_trie.fill(encoded_sentences)
    suggestions = prefix_trie.suggest((2,))
    if suggestions:
        processor.decode(suggestions[0])
        print(f"Первое продолжение для префикса 2: {processor.decode(suggestions[0])}")

    model = NGramTrieLanguageModel(encoded_sentences, 5)
    model.build()

    greedy_generator = GreedyTextGenerator(model, processor)
    beam_generator = BeamSearchTextGenerator(model, processor, 5)

    print("Жадный алгоритм до обновления:")
    greedy_result_before = greedy_generator.run(15, "Harry Potter")
    print(f"Результат: {greedy_result_before}")

    print("Beam Search до обновления:")
    beam_result_before = beam_generator.run("Harry Potter", 15)
    print(f"Результат: {beam_result_before}")

    model.update(processor.encode_sentences(ussr_letters))

    print("Жадный алгоритм после обновления:")
    greedy_result_after = greedy_generator.run(15, "Harry Potter")
    print(greedy_result_after)

    print("Beam Search (beam_width=5) (после обновления):")
    beam_result_after = beam_generator.run("Harry Potter", 15)
    print(f"Результат: {beam_result_after}")

    result = greedy_result_before
    print(f"Итоговый результат: {result}")

    print(result)
    assert result, "Result is None"


if __name__ == "__main__":
    main()
