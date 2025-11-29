"""
Generation by NGrams starter
"""

# pylint:disable=unused-import, unused-variable


def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    result = None
    assert result


if __name__ == "__main__":
    main()
