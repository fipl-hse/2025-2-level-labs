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
    result = None
    processor = TextProcessor(".")
    encoded_text = processor.encode(text)
    if encoded_text is None:
        return
    model = NGramLanguageModel(encoded_text, 7)
    model.build()
    greedy_generator = GreedyTextGenerator(model, processor)
    result_generator = greedy_generator.run(51, "Vernon")
    print(result_generator)
    beam_search = BeamSearchTextGenerator(model, processor, 3)
    beam_search_ = beam_search.run("Vernon", 56)
    result_beam = beam_search_
    print(result_beam)
    result = None

if __name__ == "__main__":
    main()
