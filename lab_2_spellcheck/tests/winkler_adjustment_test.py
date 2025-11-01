"""
Checks the second lab Winkler adjustment function
"""

import unittest

import pytest

from config.constants import FLOAT_TOLERANCE
from lab_2_spellcheck.main import winkler_adjustment


class WinklerAdjustmentDistanceTest(unittest.TestCase):
    """
    Tests function for Winkler adjustment calculation.
    """

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark10
    def test_winkler_adjustment_ideal(self) -> None:
        """
        Ideal scenario
        """
        self.assertAlmostEqual(
            winkler_adjustment("match", "maych", jaro_distance=0.1333), 0.0267, FLOAT_TOLERANCE
        )

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark10
    def test_winkler_adjustment_whole_word(self) -> None:
        """
        Ideal scenario
        """
        self.assertAlmostEqual(
            winkler_adjustment("word", "word", jaro_distance=0.0), 0.0, FLOAT_TOLERANCE
        )

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark10
    def test_winkler_adjustment_no_prefix(self) -> None:
        """
        Ideal scenario
        """
        self.assertAlmostEqual(
            winkler_adjustment("word", "ord", jaro_distance=0.0833), 0.0, FLOAT_TOLERANCE
        )

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark10
    def test_winkler_adjustment_bad_input(self) -> None:
        """
        Bad input argument scenario
        """
        bad_words = [[], {}, (), None, 42, 3.14, True]
        bad_floats = [None, True, 42, [], {}, "", ()]

        for bad_word in bad_words:
            self.assertIsNone(winkler_adjustment(bad_word, "word", 0.0, 0.1))
            self.assertIsNone(winkler_adjustment("word", bad_word, 0.0, 0.1))
            for bad_float in bad_floats:
                self.assertIsNone(winkler_adjustment("word", "word", bad_float, 0.1))
                self.assertIsNone(winkler_adjustment("word", "word", 0.0, bad_float))

                self.assertIsNone(winkler_adjustment(bad_word, bad_word, bad_float, bad_float))

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark10
    def test_winkler_adjustment_value_check(self) -> None:
        """
        Check returned value
        """
        self.assertIsInstance(winkler_adjustment("word", "word", 0.0), float)
