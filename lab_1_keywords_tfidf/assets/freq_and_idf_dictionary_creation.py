"""
Example of frequency and IDF dictionaries creation.
"""

import json
import shutil
import subprocess
import zipfile
from collections import Counter
from functools import reduce
from math import log
from pathlib import Path

from spacy.tokens import Token

from config.cli_unifier import choose_python_exe

ASSETS_PATH = Path(__file__).parent
ZIP_FILE = ASSETS_PATH / "fairy_tales.zip"
TEXTS_FOLDER = ASSETS_PATH / "fairy_tales"
FREQUENCY_PATH = ASSETS_PATH / "corpus_frequencies.json"
IDF_PATH = ASSETS_PATH / "IDF.json"


# tokenize texts
def token_is_valid(token: Token) -> bool:
    """
    Check if the token is valid.

    Args:
        token (Token): Token.

    Returns:
        bool: True if token is a word, otherwise False.
    """
    return not (token.is_stop or token.is_space or token.is_punct)


def tokenize(text: str) -> list[str]:
    """
    Tokenize the text.

    Args:
        text (str): Text to tokenize.

    Returns:
        list[str]: List of tokens.
    """
    return [token.text.lower() for token in NLP(text) if token_is_valid(token)]


def extend_tokens(all_tokens: list[str], tokens: list[str]) -> list[str]:
    """
    Extend tokens list.

    Args:
        all_tokens (list[str]): List to extend.
        tokens (list[str]): List to add.

    Returns:
        list[str]: Extended list of tokens.
    """
    return all_tokens + tokens


def main() -> None:
    """
    Creates the dictionaries.
    """
    # unpack text files
    with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
        zip_ref.extractall(TEXTS_FOLDER)

    # read the texts
    texts = []
    for tale in TEXTS_FOLDER.iterdir():
        with open(tale, "r", encoding="utf-8") as file:
            texts.append(file.read())

    tokenized_texts = list(map(tokenize, texts))
    all_tokens = reduce(extend_tokens, tokenized_texts)

    # create frequency dict
    frequency_dict = Counter(all_tokens)
    with open(FREQUENCY_PATH, "w", encoding="utf-8") as file:
        json.dump(frequency_dict, file, ensure_ascii=False)

    # create IDF dict
    unique_tokens = [list(set(tokens)) for tokens in tokenized_texts]
    unique_occurrences = reduce(extend_tokens, unique_tokens)
    n_including_docs = Counter(unique_occurrences)
    idf = {key: log(len(texts) / (value + 1)) for key, value in n_including_docs.items()}
    with open(IDF_PATH, "w", encoding="utf-8") as file:
        json.dump(idf, file, ensure_ascii=False)

    # clean up
    shutil.rmtree(TEXTS_FOLDER)

    print("Everything is generated!")


if __name__ == "__main__":
    try:
        import spacy

        subprocess.run(
            [choose_python_exe(), "-m", "spacy", "download", "ru_core_news_sm"], check=True
        )
        NLP = spacy.load("ru_core_news_sm")
        main()
    except ModuleNotFoundError as exc:
        print("Couldn't download the model.")
