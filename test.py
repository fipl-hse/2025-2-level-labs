def get_top_n(frequencies: dict[str, int | float], top: int) -> list[str] | None:
    """
    Extract the most frequent tokens.

    Args:

        frequencies (dict[str, int | float]): A dictionary with tokens and their frequencies
        top (int): Number of tokens to extract

    Returns:
        list[str] | None: Top-N tokens sorted by frequency.
        In case of corrupt input arguments, None is returned.
    """
    if isinstance(frequencies, dict):
        tempo_dict = frequencies
        top_list = []
        while len(top_list) != top:
            top_word = max(tempo_dict, key=tempo_dict.get)
            top_list.append(top_word)
            tempo_dict.pop(top_word)
        return top_list
a = 10
print(get_top_n(a, 10))