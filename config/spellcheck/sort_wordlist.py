"""
Module to check if wordlist is properly sorted.
"""

import re
from pathlib import Path

from config.console_logging import get_child_logger

logger = get_child_logger(__file__)


def check_wordlist(wordlist_path: Path) -> None:
    """
    Check that wordlist is properly sorted.

    Args:
        wordlist_path (Path): Path to wordlist
    """
    with open(wordlist_path, encoding="utf-8") as f:
        original_text = f.read()
        words = [i.strip().lower() for i in original_text.split("\n") if i.strip()]

    russian_letters_pattern = re.compile(r"[а-я]+", re.IGNORECASE)
    russian_words = [i for i in words if russian_letters_pattern.match(i)]
    english_words = list(set(words) - set(russian_words))
    new_content = "\n".join(
        sorted(russian_words)
        + sorted(english_words)
        + [
            "",
        ]
    )

    are_same = original_text == new_content
    logger.info(f"Wordlist {wordlist_path} is sorted well: {are_same}")

    if are_same:
        return

    logger.info(f"Writing new content for {wordlist_path}")
    with open(wordlist_path, "w", encoding="utf-8") as f:
        f.write(new_content)


def main() -> None:
    """
    Call functions.
    """
    main_wordlist_path = Path(__file__).parent / ".wordlist.txt"
    secondary_wordlist_path = Path(__file__).parent / ".wordlist_en.txt"

    for current_path in (main_wordlist_path, secondary_wordlist_path):
        if current_path.exists():
            check_wordlist(current_path)


if __name__ == "__main__":
    main()
