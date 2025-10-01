"""
Checks the second lab Jaro distance calculation function
"""

# pylint: disable=duplicate-code

import unittest

import pytest

from config.constants import FLOAT_TOLERANCE
from lab_2_spellcheck.main import calculate_jaro_distance


class CalculateJaroDistanceTest(unittest.TestCase):
    """
    Tests function for Jaro distance calculation.
    """

    def setUp(self):
        self.matches = {
            "word": (4, [True, True, True, True], [True, True, True, True]),
            "ord": (3, [True, True, True], [False, True, True, True]),
            "cord": (3, [False, True, True, True], [False, True, True, True]),
            "board": (3, [False, True, False, True, True], [False, True, True, True]),
            "owrd": (4, [True, True, True, True], [True, True, True, True]),
            "drow": (4, [True, True, True, True], [True, True, True, True]),
            "different": (
                2,
                [True, False, False, False, False, True, False, False, False],
                [False, False, True, True],
            ),
        }
        self.transpositions = [0, 0, 0, 0, 1, 2, 1]

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark10
    def test_calculate_jaro_distance_ideal(self):
        """
        Ideal scenario
        """
        expected_values = [0.0, 0.0833, 0.1667, 0.2167, 0.0833, 0.1667, 0.5926]
        for match, transpositions_count, expected in zip(
            self.matches.items(), self.transpositions, expected_values
        ):
            misspelled_token, match_stats = match
            match_count, _, _ = match_stats
            value = calculate_jaro_distance(
                misspelled_token, "word", match_count, transpositions_count
            )
            self.assertAlmostEqual(value, expected, FLOAT_TOLERANCE)

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark10
    def test_calculate_jaro_distance_bad_input(self):
        """
        Bad input argument scenario
        """
        bad_words = [[], {}, (), None, 42, 3.14, True]
        bad_counts = [None, 3.14, -1, [], {}, "", ()]
        for bad_word in bad_words:
            self.assertIsNone(calculate_jaro_distance(bad_word, "word", 1, 1))
            self.assertIsNone(calculate_jaro_distance("word", bad_word, 1, 1))
            for bad_count in bad_counts:
                self.assertIsNone(calculate_jaro_distance("park", "apes", bad_count, 1))
                self.assertIsNone(calculate_jaro_distance("word", "wasp", 1, bad_count))

                self.assertIsNone(calculate_jaro_distance(bad_word, bad_word, bad_count, bad_count))

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark10
    def test_calculate_jaro_distance_value_check(self):
        """
        Check returned value
        """
        self.assertIsInstance(calculate_jaro_distance("word", "wasp", 1, 0), float)

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark10
    def test_calculate_jaro_distance_zero_matches(self):
        """
        Zero matches scenario
        """
        self.assertAlmostEqual(calculate_jaro_distance("ant", "fir", 0, 0), 1.0, FLOAT_TOLERANCE)
