"""
Generation by NGrams starter
"""

# pylint:disable=unused-import, unused-variable

import os

from lab_3_generate_by_ngrams.main import (
    BackOffGenerator,
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
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, "assets", "Harry_Potter.txt")

    with open(file_path, "r", encoding="utf-8") as text_file:
        text = text_file.read()

    processor = TextProcessor('_')
    encoded_text = processor.encode(text)
    if not encoded_text:
        return

    n_gram_models = {}
    for n in range(14):
        model = NGramLanguageModel(encoded_text, n)
        model.build()
        n_gram_models[n] = model

    greedy_output = GreedyTextGenerator(
        n_gram_models[7], processor).run(51, "Vernon")
    beam_output = BeamSearchTextGenerator(
        n_gram_models[7], processor, beam_width=5
    ).run("Vernon", 51)
    backoff_output = BackOffGenerator(
        tuple(n_gram_models.values()),
        processor
    ).run(51, "Vernon")

    result = backoff_output
    assert result

    print(f"Greedy: {greedy_output}")
    print(f"Beam: {beam_output}")
    print(f"BackOff: {backoff_output}")


if __name__ == "__main__":
    main()
