"""
Generation by NGrams starter
"""
from lab_3_generate_by_ngrams.main import (
    TextProcessor,
    NGramLanguageModel,
    GreedyTextGenerator,
    BeamSearchTextGenerator,
)
# pylint:disable=unused-import, unused-variable


def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()

    # text_test = "She is happy. He is happy"
    # Шаг 1.10. Продемонстрировать результаты в start.py

    # text_processor = TextProcessor("_")
    # encoded = text_processor.encode(text)
    # language_model = NGramLanguageModel(encoded, 7)
    # language_model.build()
    # # print(encoded)
    # greedy_generator = GreedyTextGenerator(language_model, text_processor)
    # print(greedy_generator.run(51, "Vernon"))

    text_processor = TextProcessor("_")
    encoded_text = text_processor.encode(text) or None
    language_model = NGramLanguageModel(encoded_text, 7)
    if language_model.build():
        return None

    greedy_text_generator = GreedyTextGenerator(language_model, text_processor)
    beam_text_generator = BeamSearchTextGenerator(language_model, text_processor, beam_width=3)

    greedy_generated = greedy_text_generator.run(51, "Vernon") or None
    beam_generated = beam_text_generator.run("Vernon", 56) or None
    print(f"Generated with greedy algorithm: '{greedy_generated}'")
    print(f"Generated with beam algorithm: '{beam_generated}'")


if __name__ == "__main__":
    main()
