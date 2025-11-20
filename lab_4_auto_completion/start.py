"""
Auto-completion start
"""

# pylint:disable=unused-variable
from lab_3_generate_by_ngrams.main import BeamSearchTextGenerator, NGramLanguageModel, TextProcessor


def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/hp_letters.txt", "r", encoding="utf-8") as letters_file:
        hp_letters = letters_file.read()
    with open("./assets/ussr_letters.txt", "r", encoding="utf-8") as text_file:
        ussr_letters = text_file.read()

    with open("./assets/secrets/secret_4.txt", "r", encoding="utf-8") as text_file:
        letter = text_file.read()
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()

    processor = TextProcessor(end_of_word_token=".")
    encoded_text = processor.encode(text)

    n_gram_size = 18
    beam_width = 3
    seq_len = 128

    model = NGramLanguageModel(encoded_text, n_gram_size)
    model.build()

    generator = BeamSearchTextGenerator(model, processor, beam_width)

    context = letter.split("<BURNED>")[0]
    prompt = context[-n_gram_size:]
    burned_text = generator.run(prompt, seq_len)

    cleaned_text = burned_text[len(prompt):].strip()

    completed_letter = letter.replace("<BURNED>", cleaned_text)
    print(f'\n{completed_letter}')
    result = completed_letter
    assert result, "Result is None"

if __name__ == "__main__":
    main()
