"""
Auto-completion start
"""

from lab_3_generate_by_ngrams.main import BeamSearcher, BeamSearchTextGenerator, NGramLanguageModel
from lab_4_auto_completion.main import WordProcessor

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
    with open("./assets/secrets/secret_4.txt", "r", encoding="utf-8") as text_file:
        secret_4 = text_file.read()
    #burned_text_start = secret_4.find("<BURNED>")
    word_processor = WordProcessor(".")
    encoded_corpus = word_processor.encode_sentences(hp_letters)
    encoded_secret_4 = word_processor.encode_sentences(secret_4)
    burned_start_id = word_processor.get_id("<")
    for index, sentence in enumerate(encoded_secret_4):
        if burned_start_id in sentence:
            burned_sentence_index = index 
    language_model = NGramLanguageModel(encoded_corpus, 2)
    language_model.build()
    context = encoded_secret_4[burned_sentence_index - 1:burned_sentence_index + 1]
    #beam_searcher = BeamSearcher(7, language_model)
    text_generator = BeamSearchTextGenerator(language_model, word_processor, 7)
    result = text_generator.run(context, 10)
    print(result)
    assert result, "Result is None"


if __name__ == "__main__":
    main()
