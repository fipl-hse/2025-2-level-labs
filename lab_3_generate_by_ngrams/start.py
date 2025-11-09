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
    text_processor = TextProcessor("_")
    encoded = text_processor.encode(text) or None
    model = NGramLanguageModel(encoded, 7)
    if model.build():
        return None
    greedy_generator = GreedyTextGenerator(model, text_processor)
    beam_generator = BeamSearchTextGenerator(model, text_processor, beam_width=3)
    greedy= greedy_generator.run(51, "Vernon") or None
    beam = beam_generator.run("Vernon", 56) or None
    print(greedy)
    print(beam)
    result = beam
    assert result


if __name__ == "__main__":
    main()
