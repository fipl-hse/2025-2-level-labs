# pylint: disable=protected-access
"""
Checks the third lab's NGramLanguageModel class
"""
import unittest
from unittest import mock

import pytest

from lab_3_generate_by_ngrams.main import NGramLanguageModel, TextProcessor


class NGramLanguageModelTest(unittest.TestCase):
    """
    Tests NGramLanguageModel class functionality
    """

    def setUp(self) -> None:
        """
        Setup for NGramLanguageModelTest
        """
        self.text_processor = TextProcessor("_")
        self.encoded = self.text_processor.encode(
            "Text Generation is the task of generating text"
            " with the goal of appearing indistinguishable "
            "to human-written text."
        )
        self.extracted_ngrams = (
            (1, 2),
            (2, 3),
            (3, 1),
            (1, 0),
            (0, 4),
            (4, 2),
            (2, 5),
            (5, 2),
            (2, 6),
            (6, 7),
            (7, 1),
            (1, 8),
            (8, 9),
            (9, 5),
            (5, 0),
            (0, 8),
            (8, 10),
            (10, 0),
            (0, 1),
            (1, 11),
            (11, 2),
            (2, 0),
            (0, 1),
            (1, 7),
            (7, 10),
            (10, 12),
            (12, 0),
            (0, 9),
            (9, 13),
            (13, 0),
            (0, 4),
            (4, 2),
            (2, 5),
            (5, 2),
            (2, 6),
            (6, 7),
            (7, 1),
            (1, 8),
            (8, 5),
            (5, 4),
            (4, 0),
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 1),
            (1, 0),
            (0, 14),
            (14, 8),
            (8, 1),
            (1, 11),
            (11, 0),
            (0, 1),
            (1, 11),
            (11, 2),
            (2, 0),
            (0, 4),
            (4, 9),
            (9, 7),
            (7, 15),
            (15, 0),
            (0, 9),
            (9, 13),
            (13, 0),
            (0, 7),
            (7, 16),
            (16, 16),
            (16, 2),
            (2, 7),
            (7, 6),
            (6, 8),
            (8, 5),
            (5, 4),
            (4, 0),
            (0, 8),
            (8, 5),
            (5, 17),
            (17, 8),
            (8, 10),
            (10, 1),
            (1, 8),
            (8, 5),
            (5, 4),
            (4, 18),
            (18, 8),
            (8, 10),
            (10, 11),
            (11, 7),
            (7, 19),
            (19, 15),
            (15, 2),
            (2, 0),
            (0, 1),
            (1, 9),
            (9, 0),
            (0, 11),
            (11, 18),
            (18, 20),
            (20, 7),
            (7, 5),
            (5, 14),
            (14, 6),
            (6, 8),
            (8, 1),
            (1, 1),
            (1, 2),
            (2, 5),
            (5, 0),
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 1),
            (1, 0),
        )

        self.ngram_frequencies = {
            (1, 2): 0.25,
            (2, 3): 0.25,
            (3, 1): 1.0,
            (1, 0): 0.1875,
            (0, 4): 0.1875,
            (4, 2): 0.3333,
            (2, 5): 0.25,
            (5, 2): 0.2222,
            (2, 6): 0.1667,
            (6, 7): 0.5,
            (7, 1): 0.25,
            (1, 8): 0.1875,
            (8, 9): 0.1,
            (9, 5): 0.2,
            (5, 0): 0.2222,
            (0, 8): 0.125,
            (8, 10): 0.3,
            (10, 0): 0.25,
            (0, 1): 0.375,
            (1, 11): 0.1875,
            (11, 2): 0.4,
            (2, 0): 0.25,
            (1, 7): 0.0625,
            (7, 10): 0.125,
            (10, 12): 0.25,
            (12, 0): 1.0,
            (0, 9): 0.125,
            (9, 13): 0.4,
            (13, 0): 1.0,
            (8, 5): 0.4,
            (5, 4): 0.3333,
            (4, 0): 0.3333,
            (0, 14): 0.0625,
            (14, 8): 0.5,
            (8, 1): 0.2,
            (11, 0): 0.2,
            (4, 9): 0.1667,
            (9, 7): 0.2,
            (7, 15): 0.125,
            (15, 0): 0.5,
            (0, 7): 0.0625,
            (7, 16): 0.125,
            (16, 16): 0.5,
            (16, 2): 0.5,
            (2, 7): 0.0833,
            (7, 6): 0.125,
            (6, 8): 0.5,
            (5, 17): 0.1111,
            (17, 8): 1.0,
            (10, 1): 0.25,
            (4, 18): 0.1667,
            (18, 8): 0.5,
            (10, 11): 0.25,
            (11, 7): 0.2,
            (7, 19): 0.125,
            (19, 15): 1.0,
            (15, 2): 0.5,
            (1, 9): 0.0625,
            (9, 0): 0.2,
            (0, 11): 0.0625,
            (11, 18): 0.2,
            (18, 20): 0.5,
            (20, 7): 1.0,
            (7, 5): 0.125,
            (5, 14): 0.1111,
            (14, 6): 0.5,
            (1, 1): 0.0625,
        }

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_ngram_language_model_fields(self) -> None:
        """
        Checks if NGramLanguageModel fields are created correctly
        """
        language_model = NGramLanguageModel(self.encoded, 2)
        self.assertEqual(2, language_model._n_gram_size)

        language_model.build()
        actual = language_model._n_gram_frequencies
        self.assertEqual(self.ngram_frequencies.keys(), actual.keys())
        for key in actual.keys():
            self.assertAlmostEqual(self.ngram_frequencies[key], actual[key], places=4)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark10
    def test_ngram_language_model_set_n_grams(self) -> None:
        """
        Checks NGramLanguageModel set_n_grams method ideal scenario
        """
        language_model = NGramLanguageModel(self.encoded, 2)
        language_model.set_n_grams(self.ngram_frequencies)
        self.assertEqual(self.ngram_frequencies, language_model._n_gram_frequencies)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark10
    def test_ngram_language_model_set_n_grams_invalid_input(self) -> None:
        """
        Checks NGramLanguageModel set_n_grams method with invalid inputs
        """
        language_model = NGramLanguageModel(self.encoded, 2)
        bad_inputs = [1, [None], {}, None, (), 1.1, True, "string"]
        expected = {}
        for bad_input in bad_inputs:
            language_model.set_n_grams(bad_input)
            self.assertEqual(expected, language_model._n_gram_frequencies)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_ngram_language_model_extract_n_grams(self) -> None:
        """
        Checks NGramLanguageModel _extract_n_grams method ideal scenario
        """
        language_model = NGramLanguageModel(self.encoded, 2)
        self.assertEqual(self.extracted_ngrams, language_model._extract_n_grams(self.encoded))

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_ngram_language_model_extract_n_grams_invalid_input(self) -> None:
        """
        Checks NGramLanguageModel _extract_n_grams method with invalid inputs
        """
        language_model = NGramLanguageModel(self.encoded, 2)
        bad_inputs = [1, [None], {}, None, (), 1.1, True, "string"]
        expected = None
        for bad_input in bad_inputs:
            actual = language_model._extract_n_grams(bad_input)
            self.assertEqual(expected, actual)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_ngram_language_model_extract_n_grams_return_value(self) -> None:
        """
        Checks NGramLanguageModel _extract_n_grams method return value
        """
        language_model = NGramLanguageModel(self.encoded, 2)
        actual = language_model._extract_n_grams(self.encoded)
        self.assertTrue(isinstance(actual, tuple))

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_ngram_language_model_build(self) -> None:
        """
        Checks NGramLanguageModel build method ideal scenario
        """
        language_model = NGramLanguageModel(self.encoded, 2)
        actual = language_model.build()
        self.assertEqual(0, actual)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_ngram_language_model_build_none_extract_n_grams(self) -> None:
        """
        Checks NGramLanguageModel build method with None as _extract_n_grams return value
        """
        language_model = NGramLanguageModel(self.encoded, 2)
        expected = 1
        with mock.patch.object(language_model, "_extract_n_grams", return_value=None):
            actual = language_model.build()
        self.assertEqual(expected, actual)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_ngram_language_model_build_invalid_input(self) -> None:
        """
        Checks NGramLanguageModel build method with invalid inputs
        """
        bad_inputs = [1, [None], {}, None, (), 1.1, True, "string"]
        expected = 1
        for bad_input in bad_inputs:
            language_model = NGramLanguageModel(bad_input, 2)
            actual = language_model.build()
            self.assertEqual(expected, actual)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_ngram_language_model_get_n_gram_size(self) -> None:
        """
        Checks NGramLanguageModel get_n_gram_size method ideal scenario
        """
        language_model = NGramLanguageModel(self.encoded, 2)
        self.assertEqual(2, language_model.get_n_gram_size())

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_ngram_language_model_generate_next_token(self) -> None:
        """
        Checks NGramLanguageModel generate_next_token method ideal scenario
        """
        language_model = NGramLanguageModel(self.encoded, 2)
        expected = {1: 0.375, 4: 0.1875, 9: 0.125, 8: 0.125, 14: 0.0625, 11: 0.0625, 7: 0.0625}
        language_model.build()
        actual = language_model.generate_next_token((0,))
        self.assertIsNotNone(actual)
        if actual:
            for key, value in expected.items():
                self.assertEqual(actual.keys(), expected.keys())
                self.assertAlmostEqual(value, actual[key], places=4)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_ngram_language_model_generate_next_token_invalid_input(self) -> None:
        """
        Checks NGramLanguageModel generate_next_token method with invalid inputs
        """
        language_model = NGramLanguageModel(self.encoded, 2)
        bad_inputs = [1, [None], {}, None, (), 1.1, True, "string"]
        expected = None
        for bad_input in bad_inputs:
            actual = language_model.generate_next_token(bad_input)
            self.assertEqual(expected, actual)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_ngram_language_model_generate_next_token_long_context(self) -> None:
        """
        Checks NGramLanguageModel generate_next_token method with long context
        """
        language_model = NGramLanguageModel(self.encoded, 2)
        language_model.build()
        expected = {2: 0.3333, 0: 0.3333, 18: 0.1667, 9: 0.1667}
        actual = language_model.generate_next_token(
            (
                1,
                2,
                3,
                4,
            )
        )
        self.assertIsNotNone(actual)
        if actual:
            for key, value in expected.items():
                self.assertEqual(actual.keys(), expected.keys())
                self.assertAlmostEqual(value, actual[key], places=4)
