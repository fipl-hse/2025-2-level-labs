"""
Auto-completion start
"""

# pylint:disable=unused-variable
from lab_3_generate_by_ngrams.main import BeamSearchTextGenerator, NGramLanguageModel, TextProcessor
from lab_4_auto_completion.main import TextProcessingError, WordProcessor


def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/hp_letters.txt", "r", encoding="utf-8") as letters_file:
        hp_letters = letters_file.read()
    with open("./assets/ussr_letters.txt", "r", encoding="utf-8") as text_file:
        ussr_letters = text_file.read()

    with open("./assets/secrets/secret_1.txt", "r", encoding="utf=8") as secret_file:
        secret_text = secret_file.read()
    with open("./assets/Harry_Potter.txt", "r", encoding="utf=8") as harry_file:
        harry_text = harry_file.read()

    word_processor = WordProcessor("<EoW>")

    n_gram_size = 15
    beam_width = 3
    seq_len = 100

    parts = secret_text.split("<BURNED>")
    context_before = parts[0].strip()
    context_after = parts[1]

    words = context_before.split()

    context_for_generation = " ".join(words[-n_gram_size:])

    try:
        print(context_for_generation)
        encoded_secret = word_processor.encode(harry_text)
        n_gram_model = NGramLanguageModel(tuple(encoded_secret), n_gram_size)
        n_gram_model.build()

        beam_generator = BeamSearchTextGenerator(n_gram_model, word_processor, beam_width)
        output_beam: str | None = beam_generator.run(context_for_generation, seq_len)
        recovered_letter = f"{context_before[:-n_gram_size]} {output_beam} {context_after}"

        print(recovered_letter)
        result = recovered_letter
        assert result, "Result is None"

    except TextProcessingError:
        print("Fail to process text")


if __name__ == "__main__":
    main()
