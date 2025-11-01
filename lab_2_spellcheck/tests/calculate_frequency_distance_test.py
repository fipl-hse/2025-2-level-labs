"""
Checks the second lab frequency distance calculation function
"""

# pylint: disable=duplicate-code

import unittest
from unittest import mock

import pytest

from config.constants import FLOAT_TOLERANCE
from lab_2_spellcheck.main import calculate_frequency_distance


class CalculateFrequencyDistanceTest(unittest.TestCase):
    """
    Tests function for frequency distance calculation.
    """

    def setUp(self) -> None:
        """
        Set up for frequency distance calculation tests class.
        """
        self.alphabet = list("abcdefghijklmnopqrstuvwxyz")

        self.vocabulary = {
            "35": 0.04,
            "across": 0.08,
            "boy": 0.04,
            "cat": 0.16,
            "coffee": 0.04,
            "friend": 0.04,
            "kind": 0.04,
            "library": 0.12,
            "lived": 0.04,
            "loved": 0.08,
            "named": 0.04,
            "opened": 0.04,
            "shops": 0.04,
            "smart": 0.04,
            "stories": 0.04,
            "stories101": 0.04,
            "street": 0.08,
        }

        self.empty_dict = {token: 1.0 for token in self.vocabulary}

        self.misspelled = ["boyi", "streat", "coffe", "cta"]
        self.correct_words = ["boy", "street", "coffee", "cat"]

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_frequency_distance_ideal(self) -> None:
        """
        Ideal scenario
        """
        for misspelled_word, correct_word in zip(self.misspelled, self.correct_words):
            expected_dict = dict(self.empty_dict)
            expected_dict[correct_word] = 1 - self.vocabulary.get(correct_word, 0.0)

            probable_words = calculate_frequency_distance(
                misspelled_word, self.vocabulary, self.alphabet
            )
            self.assertDictEqual(probable_words, expected_dict)
            for token, score in probable_words.items():
                self.assertAlmostEqual(score, expected_dict[token], FLOAT_TOLERANCE)

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_frequency_distance_bad_input(self) -> None:
        """
        Bad input argument scenario
        """
        bad_words = [None, True, 42, 3.14, (), {}, []]
        bad_vocabularies = [None, True, 42, 3.14, (), "document", [], {}, {"good": "bad"}, {1: 0}]
        bad_alphabets = [None, True, 42, 3.14, (), "", {}]

        for bad_word in bad_words:
            self.assertIsNone(calculate_frequency_distance(bad_word, self.vocabulary, []))
            for bad_vocabulary in bad_vocabularies:
                self.assertIsNone(calculate_frequency_distance("word", bad_vocabulary, []))
                self.assertIsNone(calculate_frequency_distance(bad_word, bad_vocabulary, []))

        for bad_alphabet in bad_alphabets:
            self.assertIsNone(calculate_frequency_distance("word", self.vocabulary, bad_alphabet))

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_frequency_distance_value_check(self) -> None:
        """
        Check returned value
        """
        actual = calculate_frequency_distance("word1", self.vocabulary, [])
        self.assertIsInstance(actual, dict)
        for key, value in actual.items():
            self.assertIsInstance(key, str)
            self.assertIsInstance(value, float)

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_frequency_distance_empty_string(self) -> None:
        """
        Check return value for the empty string input
        """
        self.assertDictEqual(
            calculate_frequency_distance("", self.vocabulary, self.alphabet), self.empty_dict
        )

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_frequency_distance_empty_alphabet(self) -> None:
        """
        Check return value for the empty string input
        """

        actual = calculate_frequency_distance("libbrary", self.vocabulary, [])
        for token, score in actual.items():
            if token == "library":
                self.assertAlmostEqual(score, 0.88, FLOAT_TOLERANCE)
            else:
                self.assertAlmostEqual(score, 1.0, FLOAT_TOLERANCE)

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_frequency_distance_propose_candidates_none(self) -> None:
        """
        Case of candidate proposal method returning None
        """
        with mock.patch("lab_2_spellcheck.main.propose_candidates", return_value=None):
            result = calculate_frequency_distance("boyi", self.vocabulary, [])

        self.assertDictEqual(self.empty_dict, result)

        for token, value in result.items():
            self.assertAlmostEqual(self.empty_dict[token], value)

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_frequency_distance_several_candidates(self) -> None:
        """
        Case of several candidates being close
        """
        expected = {
            "35": 1.0,
            "across": 1.0,
            "boy": 1.0,
            "cat": 1.0,
            "coffee": 1.0,
            "friend": 1.0,
            "kind": 1.0,
            "library": 1.0,
            "lived": 0.96,
            "loved": 0.92,
            "named": 0.96,
            "opened": 1.0,
            "shops": 1.0,
            "smart": 1.0,
            "stories": 1.0,
            "stories101": 1.0,
            "street": 1.0,
        }
        actual = calculate_frequency_distance("laved", self.vocabulary, self.alphabet)
        for token, freq in actual.items():
            self.assertAlmostEqual(expected[token], freq, FLOAT_TOLERANCE)
