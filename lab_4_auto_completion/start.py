"""
Auto-completion start
"""

# pylint:disable=unused-variable
from lab_4_auto_completion.main import WordProcessor, PrefixTrie

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

    hp_encoded_text = word_processor.encode_sentences(hp_letters)
    prefix_trie_processor.fill(hp_encoded_text)
    hp_candidates = prefix_trie_processor.suggest((2,))
    
    reverse_mapping = {v: k for k, v in word_processor._storage.items()}
    
    str_candidates = []
    for candidate in hp_candidates:
        decoded_candidate = tuple(reverse_mapping.get(word_id, "") for word_id in candidate)
        str_candidates.extend(decoded_candidate)

    result = word_processor._postprocess_decoded_text(str_candidates)
    print(result)
    assert result, "Result is None"



if __name__ == "__main__":
    main()
