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

    processor_hp = TextProcessor(end_of_word_token="_")
    encoded_text = processor_hp.encode(text) or tuple()

    print(processor_hp.decode(encoded_text))

    model = NGramLanguageModel(encoded_text, 7)
    model.build()
    greedy_generator = GreedyTextGenerator(model, processor_hp).run(51, 'Vernon')
    print(f' Greedy: {greedy_generator}')

    beam_search_generator = BeamSearchTextGenerator(model, processor_hp, 3).run('Vernon', 56)
    print(f' Beam Search: {beam_search_generator}')

    reader = NGramLanguageModelReader("./assets/en_own.json", "_")
    models = []
    for ngram_size in (2, 3, 4):
        model = reader.load(ngram_size)
        if model is not None:
            models.append(model)

    backoff_processor = reader.get_text_processor()
    backoff_generator = BackOffGenerator(tuple(models), backoff_processor)

    prompts = ['Vernon said', 'The man', 'Harry', 'It is']
    best_result = None
    
    for current_prompt in prompts:
        current_result = backoff_generator.run(25, current_prompt)
        if current_result:
            words = current_result.split()
            if len(words) > 5:
                unique_words = set(words)
                repetition_ratio = len(unique_words) / len(words)
                if repetition_ratio > 0.5:
                    best_result = current_result
                    print(f'Back Off ({current_prompt}): {best_result}')
                    break
                else:
                    print(f'Back Off ({current_prompt}): {current_result} [looping]')
            else:
                best_result = current_result
                print(f'Back Off ({current_prompt}): {best_result}')
                break
        else:
            print(f'Back Off ({current_prompt}): None')

    if best_result is None and prompts:
        first_result = backoff_generator.run(15, prompts[0])
        if first_result:
            words = first_result.split()
            if len(words) > 8:
                best_result = ' '.join(words[:len(words)//2]) + '.'
            else:
                best_result = first_result
            print(f'Back Off (cropped): {best_result}')
        else:
            best_result = 'Failed'

    result = best_result
    print(f'Final Back Off: {result}')

if __name__ == "__main__":
    main()
