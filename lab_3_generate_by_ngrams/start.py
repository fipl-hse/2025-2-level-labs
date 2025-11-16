"""
Generation by NGrams starter
"""

# pylint:disable=unused-import, unused-variable
from lab_3_generate_by_ngrams.main import (
    GreedyTextGenerator,
    NGramLanguageModel,
    TextProcessor,
    BeamSearchTextGenerator,
    NGramLanguageModelReader,
    BackOffGenerator
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
    if not full_encoded_corpus:
        return
    n_gram_size = 7
    language_model = NGramLanguageModel(full_encoded_corpus, n_gram_size)
    build_result = language_model.build()
    if build_result != 0:
        return
    generated_text = GreedyTextGenerator(language_model, processor).run(51, "Vernon")
    print("3.3. Greedy algorithm generation result:")
    if generated_text:
        print(generated_text)
    else:
        print("Text generation error")
    beam_width = 3
    beam_search_generator = BeamSearchTextGenerator(language_model, processor, beam_width)
    beam_generated_text = beam_search_generator.run("Vernon", 56)
    print(f"5.4. Beam Search generation result (beam_width={beam_width}):")
    if beam_generated_text:
        print(beam_generated_text)
    else:
        print("Beam Search generation error")
    print("6-8. BackOff Generator demonstration:")
    models = []
    for n_size in [2, 3, 4, 5]:
        model = NGramLanguageModel(full_encoded_corpus, n_size)
        if model.build() == 0:
            models.append(model)
    if models:
        backoff_generator = BackOffGenerator(tuple(models), processor)
        backoff_text = backoff_generator.run(50, "The")
        print("BackOff Generator result:")
        print(backoff_text if backoff_text else "BackOff generation error")
    reader = NGramLanguageModelReader("./assets/en_own.json", "_")
    external_model = reader.load(3)
    external_processor = reader.get_text_processor()
    if external_model and external_processor:
        external_greedy = GreedyTextGenerator(external_model, external_processor)
        external_text = external_greedy.run(30, "Harry")
        print("External model generation:")
        print(external_text if external_text else "External model generation error")    
    result = external_text
    assert result


if __name__ == "__main__":
    main()
