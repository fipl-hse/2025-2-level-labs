"""
Auto-completion start
"""

# pylint:disable=unused-variable
from lab_4_auto_completion.main import (
    DynamicBackOffGenerator,
    DynamicNgramLMTrie,
    load,
    save,
    WordProcessor,
)

# def secret() -> None:
#     """
#     Launches secret solving algorithm.

#     In any case returns, None is returned
#     """
#     with open("./assets/secrets/secret_4.txt", "r", encoding="utf-8") as secret_file:
#         secret_letter = secret_file.read()
#     with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as harry_book_file:
#         harry_book_text = harry_book_file.read()
#     with open("./assets/hp_letters.txt", "r", encoding="utf-8") as harry_letters_file:
#         harry_letters_text = harry_letters_file.read()

#     n_gram_size = 2
#     beam_width = 7
#     seq_len = 10

#     processor = WordProcessor(end_of_sentence_token="<EoS>")
#     encoded_corpus = processor.encode_sentences(harry_book_text)

#     language_model = DynamicNgramLMTrie(encoded_corpus, n_gram_size)

#     language_model.clean()
#     language_model.update(encoded_corpus)

#     encoded_corpus = processor.encode_sentences(harry_letters_text)

#     language_model.update(encoded_corpus)

#     letter_parts = secret_letter.split("<BURNED>")
#     first_part = letter_parts[0].strip()
#     # second_part = letter_parts[1]

#     beam_generator = BeamSearchTextGenerator(language_model, processor, beam_width)
#     backoff_generator = DynamicBackOffGenerator(language_model, processor)

#     prompt = "".join(first_part.split()[-n_gram_size:])

#     output = beam_generator.run(prompt, seq_len)
#     print(f"beam secret: {output}")
#     output = backoff_generator.run(seq_len, prompt)
#     print(f"backoff: {output}")


def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/hp_letters.txt", "r", encoding="utf-8") as letters_file:
        hp_letters = letters_file.read()
    with open("./assets/ussr_letters.txt", "r", encoding="utf-8") as text_file:
        ussr_letters = text_file.read()


    processor = WordProcessor("<EoS>")
    encoded_corpus = processor.encode_sentences(hp_letters)

    language_model = DynamicNgramLMTrie(encoded_corpus, 5)
    language_model.build()

    path = r"C:\Users\artem\hse\2025-2-level-labs\lab_4_auto_completion\assets\dynamic_trie.json"

    save(language_model, path)
    loaded_model = load(path)


    generator = DynamicBackOffGenerator(loaded_model, processor)

    prompt = "Ivanov"

    print(f"Dynamic BackOFF with 1 corpus: {generator.run(50, prompt)}")

    encoded_corpus = processor.encode_sentences(ussr_letters)
    loaded_model.update(encoded_corpus)

    print(f"Dynamic BackOFF with 2 corpuses: {generator.run(50, prompt)}")


    result = loaded_model
    assert result


if __name__ == "__main__":
    main()
