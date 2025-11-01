"""
Checks the first lab calculate term frequencies function
"""

import unittest

import pytest

from lab_1_keywords_tfidf.main import calculate_tf


class CalculateTFTest(unittest.TestCase):
    """
    Tests calculating term frequencies function
    """

    @pytest.mark.lab_1_keywords_tfidf
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_tf_ideal(self) -> None:
        """
        Ideal calculate term frequencies scenario
        """
        frequencies = {"weather": 1, "sunny": 1, "man": 1, "happy": 1}
        expected = {"happy": 0.25, "man": 0.25, "sunny": 0.25, "weather": 0.25}

        actual = calculate_tf(frequencies)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_1_keywords_tfidf
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_tf_ideal_2(self) -> None:
        """
        Ideal calculate term frequencies scenario #2
        """
        frequencies = {
            "this": 1,
            "is": 5,
            "test": 1,
            "text": 5,
            "written": 1,
            "in": 1,
            "english": 1,
            "simple": 1,
        }
        expected = {
            "english": 0.0625,
            "in": 0.0625,
            "is": 0.3125,
            "simple": 0.0625,
            "test": 0.0625,
            "text": 0.3125,
            "this": 0.0625,
            "written": 0.0625,
        }

        actual = calculate_tf(frequencies)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_1_keywords_tfidf
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_tf_bad_input(self) -> None:
        """
        calculate term frequencies invalid input tokens check
        """
        bad_inputs = ["string", (), None, 9, 9.34, True]
        expected = None
        for bad_input in bad_inputs:
            actual = calculate_tf(bad_input)
            self.assertEqual(expected, actual)

    @pytest.mark.lab_1_keywords_tfidf
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_tf_containing_bad_input(self) -> None:
        """
        calculate term frequencies scenario with frequencies containing bad input
        """
        frequencies = {
            "this": 1,
            "is": 3,
            "test": 1,
            "text": 3,
            "written": 1,
            "in": 1,
            "english": 1,
            "simple": 1,
            None: 10,
        }
        expected = None

        actual = calculate_tf(frequencies)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_1_keywords_tfidf
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_tf_only_bad_inputs(self) -> None:
        """
        calculate term frequencies scenario with text containing only bad inputs
        """
        frequencies = {None: 10, 123: 123, (): 32}
        expected = None

        actual = calculate_tf(frequencies)
        self.assertEqual(expected, actual)
