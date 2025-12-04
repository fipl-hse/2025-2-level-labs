"""
Auto-completion start
"""

# pylint:disable=unused-variable
from lab_3_generate_by_ngrams.main import BeamSearchTextGenerator
from lab_4_auto_completion.main import NGramTrieLanguageModel, WordProcessor

def read_text(file_name: str) -> str:
    """
    Read the content from txt file from the ./assets/ directory
    """
    with open(f"./assets/{file_name}.txt", "r", encoding="utf-8") as file:
        text = file.read()
    return text

def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    # hp_letters = read_text("hp_letters")
    # ussr_letters = read_text("ussr_letters")

    secret_letter = read_text("secret_4")
    harry_book = read_text("Harry_Potter")
    harry_letters = read_text("hp_letters.txt")

    n_gram_size = 7 #2
    beam_width = 7  #7
    seq_len = 10    #10

    word_processor = WordProcessor("<EoS>")
    encoded_corpus = word_processor.encode_sentences(harry_book)

    language_model = NGramTrieLanguageModel(encoded_corpus, n_gram_size)
    print(language_model.build())
    language_model.update(word_processor.encode_sentences(harry_letters))

    prompt = "Vernon"

    encoded_prompt = word_processor.encode_sentences(prompt)
    beam_generator = BeamSearchTextGenerator(language_model, word_processor, beam_width)
    print("Start_gen")
    output = beam_generator.run("Vernon", 10)
    print(output)
    print("Stop")
    # encoded_sentences = word_processor.encode_sentences(harry_text)


    # for sentence in encoded_sentences:
    #     encoded_secret.extend(sentence)
    # encoded_secret = tuple(encoded_secret)

    # language_model = NGramLanguageModel(encoded_secret, n_gram_size)
    # language_model.build()

    # letter_parts = secret_letter.split("<BURNED>")
    # first_part = letter_parts[0].strip()
    # second_part = letter_parts[1]

    # encoded_context = word_processor.encode_sentences(first_part)
    # context = []
    # for sentence in encoded_context:
    #     context.extend(sentence)
    # context = tuple(context)

    # beam_searcher = BeamSearcher(beam_width, language_model)

    # sequence_candidates = {context: 0.0}
    # for _ in range(seq_len):
    #     new_candidates = {}
    #     for sequence, probability in sequence_candidates.items():
    #         next_tokens = beam_searcher.get_next_token(sequence)
    #         if next_tokens is None:
    #             continue
    #         updated_candidates = beam_searcher.continue_sequence(
    #             sequence, next_tokens, {sequence: probability})
    #         if updated_candidates is not None:
    #             for seq, prob in updated_candidates.items():
    #                 if seq not in new_candidates or prob < new_candidates[seq]:
    #                     new_candidates[seq] = prob
    #     if not new_candidates:
    #         break
    #     sequence_candidates = beam_searcher.prune_sequence_candidates(new_candidates) or {}
    #     if sequence_candidates is None:
    #         return None
    # if not sequence_candidates:
    #     return None

    # best_sequence = min(sequence_candidates.items(), key=lambda x: x[1])[0]
    # decoded = [word_processor.get_token(id) for id in best_sequence]
    # print(decoded)

    # result = decoded
    # assert result

if __name__ == "__main__":
    main()
