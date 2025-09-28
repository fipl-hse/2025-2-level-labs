"""
Checks the second lab Levenshtein metric function
"""

import unittest

import pytest

from lab_2_spellcheck.main import calculate_levenshtein_distance


class LevenshteinDistanceTest(unittest.TestCase):
    """
    Tests function for Levenshtein metric calculation.
    """

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_levenshtein_distance_ideal(self):
        """
        Ideal scenario
        """
        correct_word = "word"
        misspelled_words = ["word", "ord", "cord", "board", "owrd", "drow", "different"]
        expected_values = [0, 1, 1, 2, 2, 4, 8]
        for misspelled, expected in zip(misspelled_words, expected_values):
            value = calculate_levenshtein_distance(misspelled, correct_word)
            self.assertEqual(value, expected)

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_levenshtein_distance_bad_input(self):
        """
        Bad input argument scenario
        """
        bad_inputs = [[], {}, (), None, 42, 3.14, True]
        expected = None
        for bad_input in bad_inputs:
            self.assertEqual(expected, calculate_levenshtein_distance(bad_input, "word"))
            self.assertEqual(expected, calculate_levenshtein_distance("word", bad_input))
            self.assertEqual(expected, calculate_levenshtein_distance(bad_input, bad_input))

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_levenshtein_distance_value_check(self):
        """
        Check returned value
        """
        self.assertIsInstance(calculate_levenshtein_distance("word1", "word2"), int)

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_levenshtein_distance_empty_string(self):
        """
        Check return value for the empty string input
        """
        self.assertEqual(calculate_levenshtein_distance("word", ""), 4)
        self.assertEqual(calculate_levenshtein_distance("", "word"), 4)
        self.assertEqual(calculate_levenshtein_distance("", ""), 0)
