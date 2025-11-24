"""
Auto-completion start
"""

# pylint:disable=unused-variable
from lab_3_generate_by_ngrams.main import BeamSearchTextGenerator, NGramLanguageModel, TextProcessor
from lab_4_auto_completion.main import WordProcessor

def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/hp_letters.txt", "r", encoding="utf-8") as letters_file:
        hp_letters = letters_file.read()
    with open("./assets/ussr_letters.txt", "r", encoding="utf-8") as text_file:
        ussr_letters = text_file.read()

    with open("./assets/secrets/secret_1.txt", "r", encoding="utf-8") as text_file:
        letter = text_file.read()
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    word_processor = WordProcessor("<EOS>")
    encoded_sentences = word_processor.encode_sentences(text)
    all_encoded_tokens = []
    for sentence in encoded_sentences:
        all_encoded_tokens.extend(sentence)
    encoded_corpus = tuple(all_encoded_tokens)

    n_gram_size = 4
    beam_width = 3
    seq_len = 100

    model = NGramLanguageModel(encoded_corpus, n_gram_size)
    model.build()
    generator = BeamSearchTextGenerator(model, word_processor, beam_width)

    context = letter.split("<BURNED>")[0]
    encoded_context = word_processor.encode_sentences(context)
    start_length = min(len(encoded_context), n_gram_size - 1)
    prompt_tokens = encoded_context[-start_length:] if start_length > 0 else encoded_context
    print(prompt_tokens)

    prompt = word_processor.decode(tuple(prompt_tokens))
    print(prompt)
    decoded_raw = word_processor._decode(tuple(prompt_tokens))
    print("_decode result:", decoded_raw)
    # prompt = word_processor._postprocess_decoded_text(decoded_prompt)
    # print(prompt)

    burned_text = generator.run(prompt, seq_len)
    print(burned_text)
    res_letter = letter.replace("<BURNED>", cleaned_text)
    print(res_letter)
    result = res_letter
    assert result, "Result is None"


if __name__ == "__main__":
    main()
