"""
Auto-completion start
"""

# pylint:disable=unused-variable
from main import WordProcessor
from lab_3_generate_by_ngrams.main import BeamSearcher, NGramLanguageModel

def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/hp_letters.txt", "r", encoding="utf-8") as letters_file:
        hp_letters = letters_file.read()
    with open("./assets/ussr_letters.txt", "r", encoding="utf-8") as text_file:
        ussr_letters = text_file.read()
    #result = None
    #assert result, "Result is None"
    with open("./assets/secrets/secret_1.txt", "r", encoding="utf-8") as secret_file:
        secret = secret_file.read()
    print(secret)


if __name__ == "__main__":
    main()
