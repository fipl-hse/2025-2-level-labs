"""
Auto-completion start
"""

# pylint:disable=unused-variable
from lab_3_generate_by_ngrams.main import BeamSearchTextGenerator, GreedyTextGenerator, NGramLanguageModel, TextProcessor
from lab_4_auto_completion.main import WordProcessor, NGramTrieLanguageModel

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
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()

    word_processor = WordProcessor("<EOS>")
    tk = word_processor._tokenize(hp_letters)
    print(tk)
    hp_encoded_corpus = word_processor.encode(hp_letters)
    print(hp_encoded_corpus)
    model = NGramTrieLanguageModel((hp_encoded_corpus,), 3)
    build_result = model.build()
    print(build_result)

    greedy_generator = GreedyTextGenerator(model, word_processor)
    beam_generator = BeamSearchTextGenerator(model, word_processor, 3)
    test_prompt = "harry potter"
    seq_len = 52
    print(f"Text processor type: {type(greedy_generator._text_processor)}")
    print(f"Text processor EoW token: {greedy_generator._text_processor.get_end_of_word_token()}")
    greedy_result_before = greedy_generator.run(seq_len, test_prompt)
    beam_result_before = beam_generator.run(test_prompt, seq_len)

    ussr_encoded = word_processor.encode_sentences(ussr_letters)
    model.update(ussr_encoded)

    greedy_result_after = greedy_generator.run(seq_len, test_prompt)
    beam_result_after = beam_generator.run(test_prompt, seq_len)
    print(greedy_result_before)
    print(greedy_result_after)
    print(beam_result_before)
    print(beam_result_after)

    result = (greedy_result_before, beam_result_before, greedy_result_after, beam_result_after)
    assert result, "Result is None"


if __name__ == "__main__":
    main()
