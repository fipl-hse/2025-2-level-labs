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

    with open("./assets/secrets/secret_1.txt", "r", encoding="utf-8") as text_file:
        letter = text_file.read()
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    processor = TextProcessor(".")
    encoded_text = processor.encode(text)

    n_gram_size = 15
    beam_width = 3
    seq_len = 100

    model = NGramLanguageModel(encoded_text, n_gram_size)
    model.build()
    generator = BeamSearchTextGenerator(model, processor, beam_width)
    context = letter.split("<BURNED>")[0]
    prompt = context[-15:]
    #print(prompt)
    burned_text = generator.run(prompt, seq_len)
    #print(burned_text)
    cleaned_text = burned_text[len(prompt):].strip()
    print(cleaned_text)
    res_letter = letter.replace("<BURNED>", cleaned_text)
    print(res_letter)
    result = res_letter
    assert result, "Result is None"


if __name__ == "__main__":
    main()
