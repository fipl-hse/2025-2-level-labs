"""
Frequency-driven keyword extraction starter
"""

# pylint:disable=too-many-locals, unused-argument, unused-variable, invalid-name, duplicate-code
from json import load
from main import check_list, check_dict, check_positive_int, check_float, clean_and_tokenize, remove_stop_words, calculate_frequencies, get_top_n


def main() -> None:
    """
    Launches an implementation.
    """
    with open(r"C:\Users\yarak\ProgProjects\programming 2025-2026\2025-2-level-labs\lab_1_keywords_tfidf\assets\Дюймовочка.txt", "r", encoding="utf-8") as file:
        target_text = file.read()
    with open(r"C:\Users\yarak\ProgProjects\programming 2025-2026\2025-2-level-labs\lab_1_keywords_tfidf\assets\stop_words.txt", "r", encoding="utf-8") as file:
        stop_words = file.read().split("\n")
    with open(r"C:\Users\yarak\ProgProjects\programming 2025-2026\2025-2-level-labs\lab_1_keywords_tfidf\assets\IDF.json", "r", encoding="utf-8") as file:
        idf = load(file)
    with open(r"C:\Users\yarak\ProgProjects\programming 2025-2026\2025-2-level-labs\lab_1_keywords_tfidf\assets\corpus_frequencies.json", "r", encoding="utf-8") as file:
        corpus_freqs = load(file)
    result = get_top_n(calculate_frequencies(remove_stop_words(clean_and_tokenize(target_text), stop_words)), 10)
    assert result, "Keywords are not extracted"
    return result


if __name__ == "__main__":
    print(main())