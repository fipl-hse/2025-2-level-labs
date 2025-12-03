"""
Auto-completion start
"""

# pylint:disable=unused-variable
from lab_3_generate_by_ngrams.main import BeamSearchTextGenerator, GreedyTextGenerator, NGramLanguageModel
from lab_4_auto_completion.main import WordProcessor


def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/hp_letters.txt", "r", encoding="utf-8") as letters_file:
        hp_letters = letters_file.read()
    with open("./assets/ussr_letters.txt", "r", encoding="utf-8") as text_file:
        ussr_letters = text_file.read()
    with open("./assets/secrets/secret_5.txt", "r", encoding="utf-8") as secret_file:
        secret = secret_file.read()
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as hp_file:
        text = hp_file.read()
    n_gram_size = 4
    beam_width = 3
    seq_len = 25

    processor = WordProcessor('<EoS>')
    encoded_sentences = processor.encode_sentences(text)
    encoded_secret = []
    for sentence in encoded_sentences:
        encoded_secret.extend(sentence)
    encoded_secret = tuple(encoded_secret)

    model = NGramLanguageModel(encoded_secret, n_gram_size)
    print(model.build())

    letter_parts = secret.split("<BURNED>")
    before_part = letter_parts[0].strip()

    encoded_context = processor.encode_sentences(before_part)
    context = []
    for sentence in encoded_context:
        for token_id in sentence:
            token = processor.get_token(token_id)
            if token != '<EoS>':
                context.append(token_id)
    context = tuple(context)

    algorithm = BeamSearchTextGenerator(model, processor, beam_width)
    result = algorithm.run(context, seq_len)
    print(result)
    result = result
    assert result, "Result is None"

if __name__ == "__main__":
    main()
