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

class WordTextProcessor(TextProcessor):
 
        def _tokenize(self, text: str) -> tuple[str, ...] | None:
            if not isinstance(text, str) or not text:
                return None 
            text_lower = text.lower()
            tokens = []
            current_word = []
            for char in text_lower:
                if char.isalpha() or char == "'":
                    current_word.append(char)
                else:
                    if current_word:
                        word = ''.join(current_word)
                        tokens.append(word)
                        current_word = []
                    if char in '.!?':
                        tokens.append(self._end_of_word_token)
            if current_word:
                word = ''.join(current_word)
                tokens.append(word)
            if text_lower[-1] in '.!?':
                tokens.append(self._end_of_word_token)
            return tuple(tokens) if tokens else None
        
        def _put(self, element: str) -> None:
            if not isinstance(element, str) or not element:
                return
            if element not in self._storage:
                self._storage[element] = len(self._storage)

        def _postprocess_decoded_text(self, decoded_corpus: tuple[str, ...]) -> str | None:
            if not isinstance(decoded_corpus, tuple) or not decoded_corpus:
                return None
            result_parts = []
            for token in decoded_corpus:
                if token == self._end_of_word_token:
                    break
                else:
                    result_parts.append(token)
            if not result_parts:
                return None
            result_str = ' '.join(result_parts)
            result_str = result_str[0].upper() + result_str[1:]
            if not result_str.endswith('.'):
                result_str += '.'
            return result_str   

def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    processor = TextProcessor(end_of_word_token='_')
    sample_text = text[:500]
    print("1.1. Original text (first 500 characters):")
    print(sample_text)
    print("1.2. Encoding text")
    encoded_result = processor.encode(sample_text)
    print(f"Encoded result (first 30 tokens): {encoded_result[:30] if encoded_result else 'None'}")
    print("1.3. Decoded text:")
    print(processor.decode(encoded_result) if encoded_result else "Decoding error")
    full_encoded_corpus = processor.encode(text)
    if not full_encoded_corpus:
        print("Failed to encode")
    else:
        language_model = NGramLanguageModel(full_encoded_corpus, 7)
        if language_model.build() == 0:
            generated_text = GreedyTextGenerator(language_model, processor).run(51, "Vernon")
            print("3.3. Greedy algorithm generation result:")
            print(generated_text if generated_text else "Text generation error")
            print("5.4. Beam Search generation result:")
            print(BeamSearchTextGenerator(language_model, processor, 3).run("Vernon", 56) or
                "Beam Search generation error")
        print("6-8. BackOff Generator demonstration:")
        models = [model for n_size in [2, 3, 4, 5]
                if (model := NGramLanguageModel(full_encoded_corpus, n_size)).build() == 0]
        if models:
            backoff_text = BackOffGenerator(tuple(models), processor).run(50, "The")
            print("BackOff Generator result:")
            print(backoff_text if backoff_text else "BackOff generation error")
    reader = NGramLanguageModelReader("./assets/en_own.json", "_")
    external_model = reader.load(3)
    if external_model:
        external_text = GreedyTextGenerator(
            external_model, reader.get_text_processor()).run(30, "Harry")
        print("External model generation Greedy:")
        print(external_text if external_text else "External model generation error")
    print()    
    print('TASK (DEFENSE)')
    sample_text = "I love programming. I enjoy learning. You should try coding. It is fun."
    processor = WordTextProcessor(end_of_word_token='</s>')
    print("Original text")
    print(sample_text)
    encoded_result = processor.encode(sample_text)
    print(f"Encoded result: {encoded_result}")
    print("Decoded text:")
    decoded = processor.decode(encoded_result) if encoded_result else "Decoding error"
    print(decoded)
    full_encoded_corpus = processor.encode(sample_text)
    if not full_encoded_corpus:
        print("Failed to encode full corpus")
        return
    print("\nBuilding word-level language model")
    language_model = NGramLanguageModel(full_encoded_corpus, 2)
    if language_model.build() == 0:
        print("N-grams sample:", dict(list(language_model._n_gram_frequencies.items())[:5]))
        print("\nGreedy")
        greedy_generator = GreedyTextGenerator(language_model, processor)
        generated_text = greedy_generator.run(2, "i love")
        print(f"Input: 'i love'")
        print(f"Generated: {generated_text if generated_text else 'Generation error'}")
        print("\nBeam Search")
        beam_generator = BeamSearchTextGenerator(language_model, processor, 2)
        beam_text = beam_generator.run("i enjoy", 3)
        print(f"Input: 'i enjoy'")
        print(f"Generated: {beam_text if beam_text else 'Beam search error'}")
    else:
        print("Failed to build language model")
    result = beam_text
    assert result


if __name__ == "__main__":
    main()
