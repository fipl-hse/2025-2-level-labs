"""
Checks the second lab Jaro Similarity function
"""

# pylint: disable=duplicate-code

import unittest

import pytest

# from lab_2_spellcheck.main import calculate_jaro_similarity


class JaroSimilarityTest(unittest.TestCase):
    """
    Tests function for Jaro Similarity calculation.
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

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark10
    def test_calculate_jaro_similarity_ideal(self):
        """
        Ideal scenario
        """

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark10
    def test_calculate_jaro_similarity_bad_input(self):
        """
        Bad input argument scenario
        """

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark10
    def test_calculate_jaro_similarity_value_check(self):
        """
        Check returned value
        """

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_jaro_similarity_empty_string(self):
        """
        Check return value for the empty string input
        """
