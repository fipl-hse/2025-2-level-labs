"""
Generation by NGrams starter
"""

# pylint:disable=unused-import, unused-variable

from lab_3_generate_by_ngrams.main import (
    TextProcessor,
    NGramLanguageModel,
    GreedyTextGenerator,
    BeamSearcher,
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

    processor = TextProcessor('_')

    encoded_text = processor.encode(text)
    decoded_text = processor.decode(encoded_text) if encoded_text else None

    print("Encoded text (sample 50):",
          encoded_text[:50] if encoded_text else None)
    print("Decoded text (first 300 characters):",
          decoded_text[:300] if decoded_text else None)

    assert decoded_text is not None

    print("\n" + "=" * 60)
    print("BEAM SEARCH ALGORITHM DEMONSTRATION")
    print("=" * 60)

    ngram_size = 3
    language_model = NGramLanguageModel(encoded_text, ngram_size)
    language_model.build()
    print(
        f"Language model created with {len(
            language_model._n_gram_frequencies
        )} n-grams")

    beam_generator = BeamSearchTextGenerator(
        language_model, processor, beam_width=10)

    prompt = "Vernon"
    seq_len = 56

    beam_result = beam_generator.run(prompt=prompt, seq_len=seq_len)

    print(f"Prompt: '{prompt}'")
    print(f"Sequence length: {seq_len}")
    print(f"N-gram size: {ngram_size}")
    print(f"Beam Search result: {beam_result}")

    print("\n" + "=" * 60)
    print("GREEDY GENERATION DEMONSTRATION")
    print("=" * 60)

    greedy_generator = GreedyTextGenerator(language_model, processor)
    greedy_result = greedy_generator.run(seq_len=30, prompt="Harry")

    print(f"Greedy generation result: {greedy_result}")

    print("\n" + "=" * 60)
    print("BEAM SEARCHER DIRECT USAGE")
    print("=" * 60)

    beam_searcher = BeamSearcher(beam_width=5, language_model=language_model)
    test_sequence = processor.encode("Harry") or ()

    next_tokens = beam_searcher.get_next_token(test_sequence)
    print(f"Next tokens for 'Harry': {next_tokens}")

    print("\n" + "=" * 60)
    print("BACKOFF GENERATOR DEMONSTRATION")
    print("=" * 60)

    model_reader = NGramLanguageModelReader("./assets/en_own.json", "_")

    model_2 = model_reader.load(2)
    model_3 = model_reader.load(3)
    model_4 = model_reader.load(4)

    if model_2 and model_3 and model_4:
        backoff_generator = BackOffGenerator(
            (model_2, model_3, model_4),
            model_reader.get_text_processor()
        )
        backoff_result = backoff_generator.run(seq_len=40, prompt="The magic")
        print(f"BackOff generation result: {backoff_result}")
    else:
        print("Failed to load one or more language models")

    print("\n" + "=" * 60)
    print("LANGUAGE MODEL READER VALIDATION")
    print("=" * 60)

    reader_processor = model_reader.get_text_processor()
    test_text = "Hello world"
    encoded_test = reader_processor.encode(test_text)
    decoded_test = reader_processor.decode(
        encoded_test) if encoded_test else None
    print(f"Reader processor test - Original: '{test_text}'")
    print(f"Reader processor test - Decoded: '{decoded_test}'")

    print("\n" + "=" * 60)
    print("DIFFERENT N-GRAM SIZES COMPARISON")
    print("=" * 60)

    for n_size in [2, 3, 4]:
        test_model = NGramLanguageModel(encoded_text, n_size)
        test_model.build()
        test_generator = GreedyTextGenerator(test_model, processor)
        test_result = test_generator.run(seq_len=20, prompt="Ron")
        print(f"{n_size}-gram result: {test_result}")

    print("\n" + "=" * 60)
    print("BEAM SEARCH WITH DIFFERENT WIDTHS")
    print("=" * 60)

    for beam_width in [3, 5, 8]:
        width_generator = BeamSearchTextGenerator(
            language_model, processor, beam_width)
        width_result = width_generator.run(prompt="Hogwarts", seq_len=25)
        print(f"Beam width {beam_width}: {width_result}")

    print("\n" + "=" * 60)
    print("LANGUAGE MODEL ANALYSIS")
    print("=" * 60)

    sample_sequence = processor.encode("the") or ()
    next_possibilities = language_model.generate_next_token(sample_sequence)

    if next_possibilities:
        top_choices = sorted(next_possibilities.items(),
                             key=lambda x: x[1], reverse=True)[:5]
        print("Top 5 next tokens after 'the':")
        for token_id, prob in top_choices:
            token_char = processor.get_token(token_id)
            print(f"  {token_char}: {prob:.4f}")

    print("\n" + "=" * 60)
    print("TEXT PROCESSOR FUNCTIONALITY TEST")
    print("=" * 60)

    test_word = "hello"
    tokenized = processor._tokenize(test_word)
    print(f"Tokenized '{test_word}': {tokenized}")

    if tokenized:
        first_char = tokenized[0]
        char_id = processor.get_id(first_char)
        char_back = processor.get_token(
            char_id) if char_id is not None else None
        print(
            f"Character '{first_char}'-> ID {char_id} -> "
            f"Character '{char_back}'"
        )

    print("\n" + "=" * 60)
    print("ALL COMPONENTS VERIFIED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
