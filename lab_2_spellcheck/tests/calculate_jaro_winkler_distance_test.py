"""
Checks the second lab Jaro distance calculation function
"""

import unittest
from unittest import mock

import pytest

from config.constants import FLOAT_TOLERANCE
from lab_2_spellcheck.main import calculate_jaro_winkler_distance


class CalculateJaroDistanceTest(unittest.TestCase):
    """
    Tests function for Jaro distance calculation.
    """

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark10
    def test_calculate_jaro_winkler_distance_ideal(self):
        """
        Ideal scenario
        """
        self.assertAlmostEqual(
            calculate_jaro_winkler_distance("match", "maych"), 0.1067, FLOAT_TOLERANCE
        )

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark10
    def test_calculate_jaro_winkler_distance_whole_word(self):
        """
        Ideal scenario
        """
        self.assertAlmostEqual(
            calculate_jaro_winkler_distance("word", "word"), 0.0, FLOAT_TOLERANCE
        )

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark10
    def test_calculate_jaro_winkler_distance_no_prefix(self):
        """
        Ideal scenario
        """
        self.assertAlmostEqual(
            calculate_jaro_winkler_distance("word", "ord"), 0.0833, FLOAT_TOLERANCE
        )

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark10
    def test_calculate_jaro_winkler_distance_bad_input(self):
        """
        Bad input argument scenario
        """
        bad_words = [[], {}, (), None, 42, 3.14, True]
        bad_prefix_scalings = [None, 42, True, [], {}, "", ()]
        for bad_word in bad_words:
            self.assertIsNone(calculate_jaro_winkler_distance(bad_word, "word"))
            self.assertIsNone(calculate_jaro_winkler_distance("word", bad_word))
            for bad_prefix_scaling in bad_prefix_scalings:
                self.assertIsNone(
                    calculate_jaro_winkler_distance("word", "word", bad_prefix_scaling)
                )

                self.assertIsNone(
                    calculate_jaro_winkler_distance(bad_word, bad_word, bad_prefix_scaling)
                )

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark10
    def test_calculate_jaro_winkler_distance_value_check(self):
        """
        Check returned value
        """
        self.assertIsInstance(calculate_jaro_winkler_distance("word", "wasp"), float)

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark10
    def test_calculate_jaro_winkler_distance_empty_strings(self):
        """
        Scenario with empty strings
        """
        self.assertAlmostEqual(calculate_jaro_winkler_distance("", "word"), 1.0, FLOAT_TOLERANCE)
        self.assertAlmostEqual(calculate_jaro_winkler_distance("word", ""), 1.0, FLOAT_TOLERANCE)
        self.assertAlmostEqual(calculate_jaro_winkler_distance("", ""), 1.0, FLOAT_TOLERANCE)

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark10
    def test_calculate_jaro_winkler_distance_zero_matches(self):
        """
        Zero matches scenario
        """
        self.assertAlmostEqual(calculate_jaro_winkler_distance("ant", "fir"), 1.0, FLOAT_TOLERANCE)

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark10
    def test_calculate_jaro_winkler_distance_matches_none(self):
        """
        Case of matches function returning None
        """
        with mock.patch("lab_2_spellcheck.main.get_matches", return_value=None):
            result = calculate_jaro_winkler_distance("word", "ord")
        self.assertIsNone(result)

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark10
    def test_calculate_jaro_winkler_distance_transpositions_none(self):
        """
        Case of transpositions count function returning None
        """
        with mock.patch("lab_2_spellcheck.main.count_transpositions", return_value=None):
            result = calculate_jaro_winkler_distance("word", "ord")
        self.assertIsNone(result)

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark10
    def test_calculate_jaro_winkler_distance_jaro_distance_none(self):
        """
        Case of jaro distance function returning None
        """
        with mock.patch("lab_2_spellcheck.main.calculate_jaro_distance", return_value=None):
            result = calculate_jaro_winkler_distance("word", "ord")
        self.assertIsNone(result)

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark10
    def test_calculate_jaro_winkler_distance_adjustment_none(self):
        """
        Case of Winkler adjustment function returning None
        """
        with mock.patch("lab_2_spellcheck.main.winkler_adjustment", return_value=None):
            result = calculate_jaro_winkler_distance("word", "ord")
        self.assertIsNone(result)
