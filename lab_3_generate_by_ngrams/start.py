"""
Generation by NGrams starter
"""

# pylint:disable=unused-import, unused-variable
from lab_3_generate_by_ngrams.main import (
    BackOffGenerator,
    BeamSearchTextGenerator,
    GreedyTextGenerator,
    NGramLanguageModel,
    NGramLanguageModelReader,
    TextProcessor,
)

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
    processor = TextProcessor(".")
    encoded_text = processor.encode(text)
    if encoded_text is None:
        return
    model = NGramLanguageModel(encoded_text, 7)
    model.build()
    generator = GreedyTextGenerator(model, processor)
    result_generator = generator.run(51, "Vernon")
    print(result_generator)

    beam_search = BeamSearchTextGenerator(model, processor, 3)
    beam_search_ = beam_search.run("Vernon", 56)
    result_beam = beam_search_
    print(result_beam)

    language_models = []
    for n_gram_size in [1, 2, 3]:
        loaded_model = NGramLanguageModelReader("./assets/en_own.json", "_").load(n_gram_size)
        if loaded_model is not None:
            language_models.append(model)

    back_off = BackOffGenerator(tuple(language_models), processor).run(60, 'Vernon')
    result = back_off
    print(result)
    assert result

    print(f"Greedy: {greedy_output}")
    print(f"Beam: {beam_output}")
    print(f"BackOff: {backoff_output}")


if __name__ == "__main__":
    main()
