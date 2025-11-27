"""
Auto-completion start
"""

# pylint:disable=unused-variable
from lab_3_generate_by_ngrams.main import BeamSearchTextGenerator
from lab_4_auto_completion.main import NGramTrieLanguageModel, WordProcessor


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
    
    processor = WordProcessor('<EoW>')
    print("I start")
    n_gram_size = 2
    beam_width = 2
    seq_len = 5

    parts = letter.split("<BURNED>")
    part_of_letter = parts[0].strip()
    after_burned = parts[1]
    print("I did parts")

    words = part_of_letter.split()

    context_for_generation = ' '.join(words[-n_gram_size:])

    try:
        print(context_for_generation)
        encoded_text = processor.encode(text)
        ngram_model = NGramTrieLanguageModel(tuple(encoded_text), n_gram_size)
        ngram_model.build()

        result = ngram_model
        assert result, "Result is None"
    except:
        print('Failure')

if __name__ == "__main__":
    main()
