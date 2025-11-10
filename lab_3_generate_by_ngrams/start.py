"""
Generation by NGrams starter
"""
#lab_3_generate_by_ngrams.
# pylint:disable=unused-import, unused-variable
from main import (
    TextProcessor, 
    NGramLanguageModel, 
    GreedyTextGenerator)

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
    model = NGramLanguageModel(encoded_text, 7)
    build_result = model.build()
    generator = GreedyTextGenerator(model, processor)
    result = generator.run(51, "Vernon")
    print(result)
    assert result


if __name__ == "__main__":
    main()
