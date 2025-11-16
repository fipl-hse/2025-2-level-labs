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


def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    processor = TextProcessor(end_of_word_token='_')
    sample_text = text[:500]
    print("1.1. Original text (first 500 characters):")
    print(sample_text)
    print("1.2. Encoding text")
    encoded_result = processor.encode(sample_text)
    print(f"Encoded result (first 30 tokens): {encoded_result[:30] if encoded_result else 'None'}")
    decoded_result = processor.decode(encoded_result) if encoded_result else None
    print("1.3.  Decoded text:")
    print(decoded_result if decoded_result else "Decoding error")
    full_encoded_corpus = processor.encode(text)
    if full_encoded_corpus:
        language_model = NGramLanguageModel(full_encoded_corpus, 7)
        if language_model.build() == 0:
            generated_text = GreedyTextGenerator(language_model, processor).run(51, "Vernon")
            print("3.3. Greedy algorithm generation result:")
            print(generated_text if generated_text else "Text generation error")
            beam_generated_text = BeamSearchTextGenerator(language_model,
                                                          processor, 3).run("Vernon", 56)
            print("5.4. Beam Search generation result:")
            print(beam_generated_text if beam_generated_text else
                                                            "Beam Search generation error")
        print("6-8. BackOff Generator demonstration:")
        models = [model for n_size in [2, 3, 4, 5]
                 if (model := NGramLanguageModel(full_encoded_corpus, n_size)).build() == 0]
        if models:
            backoff_text = BackOffGenerator(tuple(models), processor).run(50, "The")
            print("BackOff Generator result:")
            print(backoff_text if backoff_text else "BackOff generation error")
    reader = NGramLanguageModelReader("./assets/en_own.json", "_")
    external_model = reader.load(3)
    if external_model:
        external_text = GreedyTextGenerator(
            external_model, reader.get_text_processor()).run(30, "Harry")
        print("External model generation:")
        print(external_text if external_text else "External model generation error")
    result = external_text
    assert result


if __name__ == "__main__":
    main()
