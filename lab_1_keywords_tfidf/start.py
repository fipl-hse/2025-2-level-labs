"""
Frequency-driven keyword extraction starter
"""

# from pathlib import Path
# pylint:disable=too-many-locals, unused-argument, unused-variable, invalid-name, duplicate-code
from json import load

from main import (  # calculate_chi_values,; calculate_expected_frequency,; extract_significant_words,
    calculate_frequencies,
    calculate_tf,
    calculate_tfidf,
    clean_and_tokenize,
    get_top_n,
    remove_stop_words,
)


def main() -> None:
    """
    Launches an implementation.
    """
    with open(
        r"C:\Users\polin\SuperSlay\2025-2-level-labs\lab_1_keywords_tfidf\assets\Дюймовочка.txt",
        "r",
        encoding="utf-8",
    ) as file:
        target_text = file.read()
    with open(
        r"C:\Users\polin\SuperSlay\2025-2-level-labs\lab_1_keywords_tfidf\assets\stop_words.txt",
        "r",
        encoding="utf-8",
    ) as file:
        stop_words = file.read().split("\n")
    with open(
        r"C:\Users\polin\SuperSlay\2025-2-level-labs\lab_1_keywords_tfidf\assets\IDF.json",
        "r",
        encoding="utf-8",
    ) as file:
        idf = load(file)
    with open(
        r"C:\Users\polin\SuperSlay\2025-2-level-labs\lab_1_keywords_tfidf\assets\corpus_frequencies.json",
        "r",
        encoding="utf-8",
    ) as file:
        corpus_freqs = load(file)

    tokens = clean_and_tokenize(target_text)
    wo_stop_words = remove_stop_words(tokens, stop_words)
    frequencies = calculate_frequencies(wo_stop_words)
    top_n_1 = get_top_n(frequencies, 10)
    term_frequencies = calculate_tf(frequencies)
    tf_idf = calculate_tfidf(term_frequencies, idf)
    top_n_2 = get_top_n(tf_idf, 10)
    print(top_n_2)

    #result = None
    #assert result, "Keywords are not extracted"


if __name__ == "__main__":
    main()
