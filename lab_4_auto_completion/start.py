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
    processor = WordProcessor(end_of_sentence_token="<EOS>")
    encoded_data = processor.encode_sentences(hp_letters)
    words_combined = []
    for sent in encoded_data:
        words_combined.extend(sent)
    tri_grams = []
    for idx in range(len(words_combined) - 2):
        tri_grams.append(tuple(words_combined[idx:idx + 3]))
    tree = PrefixTrie()
    tree.fill(tuple(tri_grams))
    found = tree.suggest((2,))
    print(f"Found {len(found)} suggestions for prefix (2,)")
    if found:
        best = found[0]
        print(f"First suggestion: {best}")
        output_words = []
        for code in best:
            for text, num in processor._storage.items():
                if num == code:
                    output_words.append(text)
                    break
        decoded_text = processor._postprocess_decoded_text(tuple(output_words))
        print(f"Decoded result: {decoded_text}")
    result = decoded_text
    assert result, "Result is None"


if __name__ == "__main__":
    main()
