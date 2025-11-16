"""
Generation by NGrams starter
"""

# pylint:disable=unused-import, unused-variable
from lab_3_generate_by_ngrams.main import (
    BeamSearchTextGenerator,
    GreedyTextGenerator,
    NGramLanguageModel,
    TextProcessor,
)


def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    Text = TextProcessor("_")
    encoded_text = Text.encode(text)
    print(f"\nEncoded Text: {encoded_text}")
    decoded_text = Text.decode(encoded_text)
    print(f"\nDecoded Text: {decoded_text}")
    N_Gram=NGramLanguageModel(encoded_text, n_gram_size = 7)
    N_Gram.build()
    Gr_Generator=GreedyTextGenerator(N_Gram, Text)
    Greedy_text=Gr_Generator.run(51, "Vernon")
    print(f"\nGreedy Generator: {Greedy_text}")
    Beam_Generator = BeamSearchTextGenerator(N_Gram, Text, beam_width=3)
    Beam_text = Beam_Generator.run("Vernon", 56)
    print(f"\nBeam Search Generator: {Beam_text}")
    result = Beam_text
    assert result


if __name__ == "__main__":
    main()
