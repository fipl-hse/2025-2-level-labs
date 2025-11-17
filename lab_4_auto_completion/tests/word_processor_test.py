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
        self.processor = WordProcessor(end_of_word_token="<EOW>")
        self.text = "Hello World! How are you?"

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_word_processor_tokenize_ideal(self) -> None:
        """
        Ideal _tokenize scenario.
        """
        result = self.processor._tokenize(self.text)
        self.assertIsInstance(result, tuple)
        self.assertTrue(all(isinstance(token, str) for token in result))

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_word_processor_tokenize_invalid_input(self) -> None:
        """
        Invalid inputs for WordProcessor _tokenize scenario.
        """
        bad_inputs = [1, [None], {}, None, (), 1.1, True, "", "!!!"]
        for bad_input in bad_inputs:
            with self.assertRaises(EncodingError):
                self.processor._tokenize(bad_input)

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_word_processor_postprocess_decoded_text_ideal(self) -> None:
        """
        Ideal _postprocess_decoded_text scenario.
        """
        decoded_corpus = ("hello", "world", "<EOW>", "how", "are", "you")
        result = self.processor._postprocess_decoded_text(decoded_corpus)
        self.assertIsInstance(result, str)
        self.assertTrue(result[0].isupper())
        self.assertTrue(result.endswith("."))

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_word_processor_postprocess_decoded_text_invalid_input(self) -> None:
        """
        Bad inputs for _postprocess_decoded_text scenario.
        """
        bad_inputs = [1, [None], {}, None, "", 1.1, True, ()]
        for bad_input in bad_inputs:
            with self.assertRaises(DecodingError):
                self.processor._postprocess_decoded_text(bad_input)

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_word_processor_put_ideal(self) -> None:
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
    def test_word_processor_put_invalid_input(self) -> None:
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
        decoded = ("<EOW>", "<EOW>")
        with self.assertRaises(DecodingError) as context:
            self.processor._postprocess_decoded_text(decoded)
        self.assertIn("Postprocessing resulted in empty output", str(context.exception))

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_postprocess_decoded_text_add_space_after_eow_no_duplicate(self) -> None:
        """
        Checks that _postprocess_decoded_text correctly handles EoW tokens to separate sentences.
        """
        decoded = ("hello", "world", "<EOW>", "how", "are", "you")
        result = self.processor._postprocess_decoded_text(decoded)
        self.assertEqual(result, "Hello world. How are you.")

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_word_processor_encode_sentences_ideal(self) -> None:
        """
        Ideal encode_sentences scenario.
        """
        result = self.processor.encode_sentences(self.text)
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
