"""
Auto-completion start
"""

# pylint:disable=unused-variable
from lab_3_generate_by_ngrams.main import BeamSearchTextGenerator, GreedyTextGenerator
from lab_4_auto_completion.main import NGramTrieLanguageModel, WordProcessor


def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/hp_letters.txt", "r", encoding="utf-8") as letters_file:
        hp_letters = letters_file.read()
    with open("./assets/ussr_letters.txt", "r", encoding="utf-8") as text_file:
        ussr_letters = text_file.read()

    with open("./assets/secrets/secret_4.txt", "r", encoding="utf-8") as secret_file:
        secret_letter = secret_file.read()
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as harry_book_file:
        harry_book_text = harry_book_file.read()
    with open("./assets/hp_letters.txt", "r", encoding="utf-8") as harry_letters_file:
        harry_letters_text = harry_letters_file.read()

    n_gram_size = 5 #2
    beam_width = 3
    seq_len = 51

    word_processor = WordProcessor("<EoS>")
    encoded_corpus = word_processor.encode_sentences(harry_book_text)

    language_model = NGramTrieLanguageModel(encoded_corpus, n_gram_size)
    print(language_model.build())
    language_model.update(word_processor.encode_sentences(harry_letters_text))

    prompt = "Harry Potter"

    beam_generator = BeamSearchTextGenerator(language_model, word_processor, beam_width)
    output_greedy = beam_generator.run(prompt, seq_len)

    greedy_generator = GreedyTextGenerator(language_model, word_processor)
    output_beam = greedy_generator.run(seq_len, prompt)

    print(f"Greedy: {output_greedy}")
    print(f"beam: {output_beam}")

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
