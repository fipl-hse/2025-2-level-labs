# pylint: disable=protected-access,duplicate-code
"""
Checks the DynamicBackOffGenerator class
"""
import unittest
from unittest import mock

import pytest

from config.constants import FLOAT_TOLERANCE
from lab_4_auto_completion.main import DynamicBackOffGenerator, DynamicNgramLMTrie, WordProcessor


class DynamicBackOffGeneratorTest(unittest.TestCase):
    """
    Tests DynamicBackOffGenerator class functionality
    """

    def setUp(self) -> None:
        """
        Setup of DynamicBackOffGeneratorTest.
        """
        self.encoded_corpus = (
            (1, 2),
            (1, 2, 0, 3, 4, 5),
            (3, 4, 5, 0, 6, 7, 8),
            (7, 6, 8, 0, 5, 4, 8),
            (5, 4, 8, 2),
            (1, 2, 0, 5, 4, 2),
        )
        self.max_ngram = 4
        self.model = DynamicNgramLMTrie(self.encoded_corpus, self.max_ngram)
        self.model.build()

        self.processor = WordProcessor(end_of_sentence_token="_")
        self.processor._storage = {
            "_": 0,
            "hello": 1,
            "world": 2,
            "how": 3,
            "are": 4,
            "you": 5,
            "i": 6,
            "am": 7,
            "good": 8,
        }

        self.generator = DynamicBackOffGenerator(self.model, self.processor)

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark10
    def test_fields(self) -> None:
        """
        Checks if DynamicBackOffGenerator fields are created correctly
        """
        self.assertEqual(self.processor, self.generator._text_processor)
        self.assertEqual({self.max_ngram: self.model}, self.generator._language_models)
        self.assertEqual(self.model, self.generator._dynamic_trie)

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark10
    def test_get_next_token(self) -> None:
        """
        Checks DynamicBackOffGenerator get_next_token method ideal scenario
        """
        expected = {0: 0.0625, 3: 0.0833}
        actual = self.generator.get_next_token((1, 2))
        for token, value in actual.items():
            self.assertAlmostEqual(expected[token], value, FLOAT_TOLERANCE)

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark10
    def test_get_next_token_return_value(self) -> None:
        """
        Checks DynamicBackOffGenerator get_next_token method return value
        """
        actual = self.generator.get_next_token((1, 2))
        self.assertIsInstance(actual, dict)
        for token, value in actual.items():
            self.assertIsInstance(token, int)
            self.assertIsInstance(value, float)

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark10
    def test_get_next_token_none_generate_next_token(self) -> None:
        """
        Checks DynamicBackOffGenerator get_next_token method with incorrect
        generate_next_token return value
        """
        with mock.patch.object(
            self.generator._dynamic_trie, "generate_next_token", return_value=None
        ):
            self.assertIsNone(self.generator.get_next_token((1, 2)))

        with mock.patch.object(
            self.generator._dynamic_trie, "generate_next_token", return_value={}
        ):
            self.assertIsNone(self.generator.get_next_token((1, 2)))

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark10
    def test_get_next_token_invalid_input(self) -> None:
        """
        Checks DynamicBackOffGenerator get_next_token method with invalid inputs
        """
        bad_inputs = [1, [None], {}, None, (), 1.1, True, "string"]

        for bad_input in bad_inputs:
            self.assertIsNone(self.generator.get_next_token(bad_input))

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark10
    def test_run(self) -> None:
        """
        Checks DynamicBackOffGenerator run method ideal scenario
        """
        actual = self.generator.run(10, "Hello")
        self.assertEqual("Hello world. You are good world. You are good.", actual)

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark10
    def test_run_invalid_input(self) -> None:
        """
        Checks DynamicBackOffGenerator run method with invalid inputs
        """
        generator = DynamicBackOffGenerator(self.model, self.processor)
        bad_len_inputs = [[None], {}, None, (), 1.1, "string", -1]
        bad_prompt_inputs = [1, [None], {}, None, (), 1.1, True, ""]
        expected = None
        for bad_len in bad_len_inputs:
            actual = generator.run(bad_len, "The")
            self.assertEqual(expected, actual)

        for bad_prompt in bad_prompt_inputs:
            actual = generator.run(50, bad_prompt)
            self.assertEqual(expected, actual)

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark10
    def test_run_return_value(self) -> None:
        """
        Checks DynamicBackOffGenerator run method return value
        """
        self.assertIsInstance(self.generator.run(1, "Hello"), str)

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark10
    def test_run_none_encode(self) -> None:
        """
        Checks DynamicBackOffGenerator run method with None encode return value
        """
        with mock.patch.object(self.generator._text_processor, "encode", return_value=None):
            self.assertIsNone(self.generator.run(1, "Hello"))

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark10
    def test_run_none_get_next_token(self) -> None:
        """
        Checks DynamicBackOffGenerator run method with None
        get_next_token return value
        """
        with mock.patch.object(self.generator, "get_next_token", return_value=None):
            self.assertEqual("Hello.", self.generator.run(1, "Hello"))
