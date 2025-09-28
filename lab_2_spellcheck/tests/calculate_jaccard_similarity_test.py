"""
Checks the second lab Jaccard Similarity function
"""

import unittest

import pytest

from config.constants import FLOAT_TOLERANCE
from lab_2_spellcheck.main import calculate_jaccard_distance


class JaccardSimilarityTest(unittest.TestCase):
    """
    Tests function for Jaccard Similarity calculation.
    """

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_jaccard_distance_ideal(self):
        """
        Ideal scenario
        """
        correct_word = "word"
        misspelled_words = ["word", "ord", "cord", "board", "owrd", "drow", "different"]
        expected_values = [0.0, 0.25, 0.4, 0.5, 0.0, 0.0, 0.778]
        for misspelled, expected in zip(misspelled_words, expected_values):
            value = calculate_jaccard_distance(misspelled, correct_word)
            self.assertAlmostEqual(value, expected, 3)

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_jaccard_distance_bad_input(self):
        """
        Bad input argument scenario
        """
        bad_inputs = [[], {}, (), None, 42, 3.14, True]
        for bad_input in bad_inputs:
            self.assertIsNone(calculate_jaccard_distance(bad_input, "word"))
            self.assertIsNone(calculate_jaccard_distance("word", bad_input))

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_jaccard_distance_value_check(self):
        """
        Check returned value
        """
        self.assertIsInstance(calculate_jaccard_distance("word1", "word2"), float)

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_jaccard_distance_empty_string(self):
        """
        Check return value for the empty string input
        """
        self.assertAlmostEqual(calculate_jaccard_distance("word", ""), 1.0, FLOAT_TOLERANCE)
        self.assertAlmostEqual(calculate_jaccard_distance("", "word"), 1.0, FLOAT_TOLERANCE)
        self.assertAlmostEqual(calculate_jaccard_distance("", ""), 1.0, FLOAT_TOLERANCE)
