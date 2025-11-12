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
    processor = TextProcessor(end_of_word_token='_')
    sample_text = text[:500]
    print("1.1. Original text (first 500 characters):")
    print(sample_text)
    print("1.2. Encoding text")
    encoded_result = processor.encode(sample_text)
    print(f"Encoded result (first 30 tokens): {encoded_result[:30] if encoded_result else 'None'}")
    print("1.3. Decoding text")
    decoded_result = processor.decode(encoded_result) if encoded_result else None
    print("Decoded text:")
    print(decoded_result if decoded_result else "Decoding error")
    print("1.4. Token storage information:")
    print(f"   Special token: '{processor.get_end_of_word_token()}'")
    print(f"   Special token ID: {processor.get_id('_')}")
    full_encoded_corpus = processor.encode(text)
    if not full_encoded_corpus:
        return
    n_gram_size = 7
    language_model = NGramLanguageModel(full_encoded_corpus, n_gram_size)
    build_result = language_model.build()
    if build_result != 0:
        return
    print("2.4. N-gram examples (first 5):")
    ngram_items = list(language_model._n_gram_frequencies.items())[:5]
    for i, (ngram, freq) in enumerate(ngram_items):
        decoded_ngram = []
        for token_id in ngram:
            token = processor.get_token(token_id)
            decoded_ngram.append(f"'{token}'" if token else '?')
        print(f"   {i+1}. {decoded_ngram}: {freq:.4f}")
    print("2.5. Testing next token generation")
    test_prompt = "Vernon"
    encoded_test = processor.encode(test_prompt)
    if encoded_test:
        next_tokens = language_model.generate_next_token(encoded_test)
        if next_tokens:
            print(f"   Prompt: '{test_prompt}'")
            print(f"   Possible next tokens found: {len(next_tokens)}")
            print(f"   Top-5 candidates:")
            top_tokens = list(next_tokens.items())[:5]
            for token_id, prob in top_tokens:
                token = processor.get_token(token_id)
                print(f"     '{token}': {prob:.4f}")
        else:
            print(f"   No next tokens found for prompt '{test_prompt}'")
            if len(encoded_test) >= n_gram_size - 1:
                shorter_context = encoded_test[-(n_gram_size - 1):]
                next_tokens = language_model.generate_next_token(shorter_context)
                if next_tokens:
                    print(f"   With shorter context found: {len(next_tokens)} tokens")
    text_generator = GreedyTextGenerator(language_model, processor)
    print("3.2. Generating text...")
    prompt = "Vernon"
    seq_len = 51
    print(f"Prompt: '{prompt}'")
    print(f"Sequence length: {seq_len}")
    print(f"N-gram size: {n_gram_size}")
    generated_text = text_generator.run(seq_len, prompt)
    print("3.3. Generation result:")
    if generated_text:
        print(generated_text)
    else:
        print("Text generation error")
    result = generated_text
    assert result


if __name__ == "__main__":
    main()
