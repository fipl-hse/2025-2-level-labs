"""
Auto-completion start
"""

# pylint:disable=unused-variable
from lab_4_auto_completion.main import WordProcessor, PrefixTrie
from lab_3_generate_by_ngrams.main import BeamSearcher, BeamSearchTextGenerator, TextProcessor, NGramLanguageModel

def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/hp_letters.txt", "r", encoding="utf-8") as letters_file:
        hp_letters = letters_file.read()
    with open("./assets/ussr_letters.txt", "r", encoding="utf-8") as text_file:
        ussr_letters = text_file.read()
    word_processor = WordProcessor(end_of_sentence_token="<EOS>")
    encoded = word_processor.encode_sentences(hp_letters)
    prefix_trie = PrefixTrie()
    prefix_trie.fill(encoded)
    suggestions = prefix_trie.suggest((2,))
    if suggestions:
        first_sug = suggestions[0]
        decoded = []
        for word_id in first_sug:
            for word, id in word_processor._storage.items():
               if id == word_id:
                   decoded.append(word)
                   break
        decoded_text = word_processor._postprocess_decoded_text(tuple(decoded))
        print(decoded_text)
    result = decoded_text
    assert result, "Result is None"


if __name__ == "__main__":
    main()
