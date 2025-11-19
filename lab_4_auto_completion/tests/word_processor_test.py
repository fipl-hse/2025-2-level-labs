"""
Checks Word Processor Class.
"""

# pylint: disable=protected-access

import unittest

import pytest

from lab_4_auto_completion.main import DecodingError, EncodingError, WordProcessor


class WordProcessorTest(unittest.TestCase):
    """
    Tests WordProcessor class functionality.
    """

    def setUp(self) -> None:
        """
        Setup for WordProcessorTest.
        """
        self.processor = WordProcessor(end_of_sentence_token="<EOS>")
        self.text = "Hello World! How are you?"

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_tokenize_ideal(self) -> None:
        """
        Ideal _tokenize scenario.
        """
        expected = ("hello", "world", "<EOS>", "how", "are", "you", "<EOS>")
        actual = self.processor._tokenize(self.text)
        self.assertEqual(actual, expected)
        self.assertIsInstance(actual, tuple)
        self.assertTrue(all(isinstance(token, str) for token in actual))

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_tokenize_invalid_input(self) -> None:
        """
        Invalid inputs for WordProcessor _tokenize scenario.
        """
        bad_inputs = [1, [None], {}, None, (), 1.1, True, ""]
        for bad_input in bad_inputs:
            with self.assertRaises(EncodingError) as context:
                self.processor._tokenize(bad_input)
        self.assertIn("Invalid input: text must be a non-empty string", str(context.exception))

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_word_processor_tokenize_empty_output(self) -> None:
        """
        _tokenize empty output.
        """
        test_cases = ["!!!", "123456789"]

        for text in test_cases:
            with self.assertRaises(EncodingError) as context:
                self.processor._tokenize(text)
            self.assertEqual(str(context.exception), "Tokenization resulted in empty output")

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_postprocess_decoded_text_ideal(self) -> None:
        """
        Ideal _postprocess_decoded_text scenario.
        """
        decoded_corpus = ("hello", "world", "<EOS>", "how", "are", "you")
        result = self.processor._postprocess_decoded_text(decoded_corpus)
        expected = "Hello world. How are you."
        self.assertEqual(result, expected)
        self.assertIsInstance(result, str)
        self.assertTrue(result[0].isupper())
        self.assertTrue(result.endswith("."))

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_postprocess_decoded_text_invalid_input(self) -> None:
        """
        Bad inputs for _postprocess_decoded_text scenario.
        """
        bad_inputs = [1, [None], {}, None, "", 1.1, True, ()]
        for bad_input in bad_inputs:
            with self.assertRaises(DecodingError) as context:
                self.processor._postprocess_decoded_text(bad_input)
        self.assertEqual(
            str(context.exception), "Invalid input: decoded_corpus must be a non-empty tuple"
        )

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_postprocess_decoded_text_empty_output(self) -> None:
        """
        Bad inputs for _postprocess_decoded_text scenario.
        """
        bad_inputs = [1, [None], {}, None, "", 1.1, True, ()]
        for bad_input in bad_inputs:
            with self.assertRaises(DecodingError) as context:
                self.processor._postprocess_decoded_text(bad_input)
        self.assertEqual(
            str(context.exception), "Invalid input: decoded_corpus must be a non-empty tuple"
        )

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_put_ideal(self) -> None:
        """
        Ideal _put scenario.
        """
        initial_size = len(self.processor._storage)
        self.processor._put("tea")
        self.assertEqual(len(self.processor._storage), initial_size + 1)
        self.assertIn("tea", self.processor._storage)

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_put_invalid_input(self) -> None:
        """
        Invalid inputs for _put scenario.
        """
        bad_inputs = [1, [None], {}, None, (), 1.1, True, ""]
        initial_size = len(self.processor._storage)
        for bad_input in bad_inputs:
            self.processor._put(bad_input)
        self.assertEqual(len(self.processor._storage), initial_size)

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_postprocess_decoded_text_empty_result(self) -> None:
        """
        Checks that _postprocess_decoded_text raises DecodingError when the final text is empty.
        """
        decoded = ("<EOS>", "<EOS>")
        with self.assertRaises(DecodingError) as context:
            self.processor._postprocess_decoded_text(decoded)
        self.assertIn("Postprocessing resulted in empty output", str(context.exception))

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_postprocess_decoded_text_add_space_after_eos_no_duplicate(self) -> None:
        """
        Checks that _postprocess_decoded_text correctly handles EoS tokens to separate sentences.
        """
        decoded = ("hello", "world", "<EOS>", "how", "are", "you")
        result = self.processor._postprocess_decoded_text(decoded)
        self.assertEqual(result, "Hello world. How are you.")

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_encode_sentences_ideal(self) -> None:
        """
        Ideal encode_sentences scenario.
        """
        result = self.processor.encode_sentences(self.text)
        encoded = ((1, 2, 0), (3, 4, 5, 0))
        self.assertEqual(encoded, result)
        self.assertIsInstance(result, tuple)
        self.assertTrue(all(isinstance(sentence, tuple) for sentence in result))

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_encode_sentences_return_value(self) -> None:
        """
        Check encode_sentences return value.
        """
        result = self.processor.encode_sentences(self.text)
        self.assertEqual(len(result), 2)
        self.assertGreater(len(result[0]), 0)
