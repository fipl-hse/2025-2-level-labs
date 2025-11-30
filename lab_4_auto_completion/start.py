"""
Auto-completion start
"""

from lab_3_generate_by_ngrams.main import BeamSearchTextGenerator, NGramLanguageModel, TextProcessor

# pylint:disable=unused-variable
from lab_4_auto_completion.main import PrefixTrie, WordProcessor


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
    prefix_trie_processor = PrefixTrie()

# processing secret
    # text_processor = TextProcessor("_")
    # n_gram_size = 2
    # beam_width = 7
    # seq_len = 9

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

    # beamsearcher = BeamSearchTextGenerator(n_gram_model, word_processor, beam_width)
    # burned_text = beamsearcher.run(context, seq_len)
    # print(burned_text)

    # recovered = f"{context_0} {burned_text} {context_1}"
    # print(recovered)

# end processing secret

    hp_encoded_text = word_processor.encode_sentences(hp_letters)
    prefix_trie_processor.fill(hp_encoded_text)
    hp_candidates = prefix_trie_processor.suggest((2,))
    
    reverse_mapping = {v: k for k, v in word_processor._storage.items()}
    
    str_candidates = []
    for candidate in hp_candidates:
        for word_id in candidate:
            if word_id in reverse_mapping:
                str_candidates.append(reverse_mapping[word_id])

    result = word_processor._postprocess_decoded_text(tuple(str_candidates))
    print(result)
    assert result, "Result is None"



if __name__ == "__main__":
    main()


# 