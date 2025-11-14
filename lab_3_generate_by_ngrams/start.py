"""
Generation by NGrams starter
"""

# pylint:disable=unused-import, unused-variable
from lab_3_generate_by_ngrams.main import (TextProcessor, NGramLanguageModel, GreedyTextGenerator)


def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    Text = TextProcessor("_")
    encoded_text=Text.encode(text)
    N_Gram=NGramLanguageModel(encoded_text, 7)
    N_Gram.build()
    Gr_text=GreedyTextGenerator(N_Gram, Text)
    final_text=Gr_text.run(51, "Vernon")
    print(final_text)
    result = final_text
    assert result


if __name__ == "__main__":
    main()
