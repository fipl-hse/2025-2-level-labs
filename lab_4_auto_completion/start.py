"""
Auto-completion start
"""

from lab_4_auto_completion.main import PrefixTrie, WordProcessor

# pylint:disable=unused-variable


def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/hp_letters.txt", "r", encoding="utf-8") as letters_file:
        hp_letters = letters_file.read()
    with open("./assets/ussr_letters.txt", "r", encoding="utf-8") as text_file:
        ussr_letters = text_file.read()
    result = None
    word_processor = WordProcessor(".")
    trie = PrefixTrie()
    encoded_letters = word_processor.encode_sentences(hp_letters)
    trie.fill(encoded_letters)
    suggested = trie.suggest((2,))
    first_suggested_sentence = suggested[0]
    words_list = []
    for element in first_suggested_sentence:
        token = word_processor.get_token(element)
        words_list.append(token)
    decoded_text = word_processor._postprocess_decoded_text(tuple(words_list))
    print(decoded_text)
    result = decoded_text
    assert result, "Result is None"


if __name__ == "__main__":
    main()
