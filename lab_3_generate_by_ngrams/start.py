"""
Generation by NGrams starter
"""

# pylint:disable=unused-import, unused-variable
from lab_3_generate_by_ngrams.main import (
    BeamSearchTextGenerator,
    GreedyTextGenerator,
    NGramLanguageModel,
    TextProcessor)


def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    processor = TextProcessor("_")
    encoded_text = processor.encode(text)
    if encoded_text is None:
        return None
    model = NGramLanguageModel(encoded_text[:5000], 7)
    model.build()
    generator = GreedyTextGenerator(model, processor)
    result_generator = generator.run(51, "Harry ")
    beam_search = BeamSearchTextGenerator(model, processor, 3)
    beam_search_ = beam_search.run("Harry ", 56)
    result = beam_search_
    print(result)
    assert result    


if __name__ == "__main__":
    main()
