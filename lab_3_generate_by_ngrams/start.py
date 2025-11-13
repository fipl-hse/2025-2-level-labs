"""
Generation by NGrams starter
"""

# pylint:disable=unused-import, unused-variable

from lab_3_generate_by_ngrams.main import (
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
    processor = TextProcessor(end_of_word_token='_')
    encoded_content = processor.encode(text)
    model = NGramLanguageModel(encoded_content, 7)
    model.build()
    greedy = GreedyTextGenerator(model, processor)
    greedy_result = greedy.run(51, 'Vernon')
    print("Greedy:", greedy_result)
    assert result


if __name__ == "__main__":
    main()
