"""
Auto-completion start
"""

from lab_3_generate_by_ngrams.main import BeamSearchTextGenerator, GreedyTextGenerator

# pylint:disable=unused-variable
from lab_4_auto_completion.main import NGramTrieLanguageModel, PrefixTrie, WordProcessor


def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/hp_letters.txt", "r", encoding="utf-8") as letters_file:
        hp_letters = letters_file.read()
    with open("./assets/ussr_letters.txt", "r", encoding="utf-8") as text_file:
        ussr_letters = text_file.read()

    word_processor = WordProcessor("_")
    # prefix_trie_processor = PrefixTrie()

# processing secret
    # text_processor = TextProcessor("_")
    # n_gram_size = 2
    beam_width = 7
    seq_len = 9

    # with open("./assets/secrets/secret_1.txt", "r", encoding="utf-8") as secret_file:
    #     secret_text = secret_file.read()

    # context_0, context_1 = secret_text.split("<BURNED>")
    # words = context_0.split()
    # context_words = words[-min(3, len(words)):]
    # context = " ".join(context_words)

    # full_secret_text = context_0 + " " + context_1
    # encoded_secret_text = text_processor.encode(full_secret_text)

    # encoded_secret_text = word_processor.encode(secret_text)
    # n_gram_model = NGramLanguageModel(encoded_secret_text, n_gram_size)
    # n_gram_model.build()


    # burned_text = beamsearcher.run(context, seq_len)
    # print(burned_text)

    # recovered = f"{context_0} {burned_text} {context_1}"
    # print(recovered)

# end processing secret

    # prefix_trie_processor.fill(hp_encoded_text)
    # hp_candidates = prefix_trie_processor.suggest((2,))

    # reverse_mapping = {v: k for k, v in word_processor._storage.items()}

    # str_candidates = []
    # for candidate in hp_candidates:
    #     for word_id in candidate:
    #         if word_id in reverse_mapping:
    #             str_candidates.append(reverse_mapping[word_id])



    hp_encoded_text = word_processor.encode_sentences(hp_letters)
    ussr_encoded_text = word_processor.encode_sentences(ussr_letters)

    ngram_trie_lm = NGramTrieLanguageModel(hp_encoded_text, 5)
    ngram_trie_lm.build()

    greedy_generator = GreedyTextGenerator(ngram_trie_lm, word_processor)
    ussr_corp_greedy = greedy_generator.run(seq_len, ussr_letters)
    print("Greedy before update", ussr_corp_greedy)
    print()

    ngram_trie_lm.update(ussr_encoded_text)
    ussr_corp_greedy = greedy_generator.run(seq_len, ussr_letters)
    print("Greedy aftert update", ussr_corp_greedy)
    print()

    ngram_trie_lm = NGramTrieLanguageModel(hp_encoded_text, 5)
    ngram_trie_lm.build()

    beamsearcher = BeamSearchTextGenerator(ngram_trie_lm, word_processor, beam_width)
    ussr_corp_beam = beamsearcher.run(ussr_letters, seq_len)
    print("Beamsearcher before update", ussr_corp_beam)
    print()

    ngram_trie_lm.update(ussr_encoded_text)
    ussr_corp_beam = beamsearcher.run(ussr_letters, seq_len)
    print("Beamsearcher after update", ussr_corp_beam)
    print()
    result = ussr_corp_beam
    print(result)
    assert result, "Result is None"


if __name__ == "__main__":
    main()
