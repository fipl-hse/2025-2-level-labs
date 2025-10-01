"""
Checks the second lab frequency distance calculation function
"""

# pylint: disable=duplicate-code

import unittest

import pytest

from config.constants import FLOAT_TOLERANCE
from lab_2_spellcheck.main import calculate_frequency_distance


class CalculateFrequencyDistanceTest(unittest.TestCase):
    """
    Tests function for frequency distance calculation.
    """

    def setUp(self) -> None:
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
    def test_calculate_frequency_distance_ideal(self):
        """
        Ideal scenario
        """
        for misspelled_word, correct_word in zip(self.misspelled, self.correct_words):
            expected_dict = dict(self.empty_dict)
            expected_dict[correct_word] = self.vocabulary.get(correct_word, 0.0)

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
    def test_calculate_frequency_distance_bad_input(self):
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
    def test_calculate_frequency_distance_value_check(self):
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
    def test_calculate_frequency_distance_empty_string(self):
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
    def test_calculate_frequency_distance_empty_alphabet(self):
        """
        Check return value for the empty string input
        """

        actual = calculate_frequency_distance("libbrary", self.vocabulary, [])
        for token, score in actual.items():
            if token == "library":
                self.assertAlmostEqual(score, 0.12, FLOAT_TOLERANCE)
            else:
                self.assertAlmostEqual(score, 1.0, FLOAT_TOLERANCE)
