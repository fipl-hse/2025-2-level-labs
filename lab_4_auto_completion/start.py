"""
Auto-completion start
"""

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
    word_processor = WordProcessor(".")
    prefix_trie = PrefixTrie()
    encoded_letters = word_processor.encode_sentences(ussr_letters)
    prefix_trie.fill(encoded_letters)
    suggestion = prefix_trie.suggest((2,))
    first_suggestion = suggestion[0]
    words_list = [word_processor.get_token(element) for element in first_suggestion]
    decoded = word_processor._postprocess_decoded_text(tuple(words_list))
    print(decoded)
    result = decoded
    assert result, "Result is None"

if __name__ == "__main__":
    main()
