"""
Generation by NGrams starter
"""

# pylint:disable=unused-import, unused-variable
from lab_3_generate_by_ngrams.main import (
    BackOffGenerator,
    BeamSearchTextGenerator,
    GreedyTextGenerator,
    NGramLanguageModel,
    NGramLanguageModelReader,
    TextProcessor,
)


def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    processor = TextProcessor(".")
    encoded_text = processor.encode(text)
    if encoded_text is None:
        return
    model = NGramLanguageModel(encoded_text, 7)
    model.build()
    generator = GreedyTextGenerator(model, processor)
    result_generator = generator.run(51, "Vernon")
    print(result_generator)

    beam_search = BeamSearchTextGenerator(model, processor, 3)
    beam_search_ = beam_search.run("Vernon", 56)
    result_beam = beam_search_
    print(result_beam)

    language_models = []
    for n_gram_size in [1, 2, 3]:
        loaded_model = NGramLanguageModelReader("./assets/en_own.json", "_").load(n_gram_size)
        if loaded_model is not None:
            language_models.append(model)

    back_off = BackOffGenerator(tuple(language_models), processor).run(60, 'Vernon')
    result = back_off
    print(result)
    assert result

    processor = TextProcessor(end_of_word_token="_")
    encoded = processor.encode(text) or tuple()
    print(processor.decode(encoded))

    model = NGramLanguageModel(encoded, 7)
    model.build()

    print(f'Greedy: {GreedyTextGenerator(model, processor).run(51, "Vernon")}')
    print(f'Beam Search: {BeamSearchTextGenerator(model, processor, 3).run("Vernon", 56)}')

    reader = NGramLanguageModelReader("./assets/en_own.json", "_")
    models = []
    for n_size in (2, 3, 4):
        loaded = reader.load(n_size)
        if loaded is not None:
            models.append(loaded)

    generator = BackOffGenerator(tuple(models), reader.get_text_processor())
    prompts = ['Vernon said', 'The man', 'Harry', 'It is']
    result = None
    for prompt in prompts:
        current = generator.run(15, prompt)
        if not current:
            continue
        word_list = current.split()
        if len(word_list) > 5 and len(set(word_list)) / len(word_list) > 0.5:
            result = current
            print(f'Back Off ({prompt}): {result}')
            break
        if len(word_list) <= 5:
            result = current
            print(f'Back Off ({prompt}): {result}')
            break
        print(f'Back Off ({prompt}): {current} [looping]')
    if result is None and prompts:
        current = generator.run(10, prompts[0])
        if current:
            word_list = current.split()
            result = ' '.join(word_list[:len(word_list)//2]) + '.' if len(word_list)>8 else current
            print(f'Back Off (cropped): {result}')
    assert result
    print(f'Final Back Off: {result}')

if __name__ == "__main__":
    main()
