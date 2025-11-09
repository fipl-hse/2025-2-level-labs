# pylint: disable=duplicate-code
"""
Checks the second lab filling of Levenshtein metric matrix
"""

import unittest
from unittest import mock

import pytest

from lab_2_spellcheck.main import fill_levenshtein_matrix


class FillLevenshteinMatrixTest(unittest.TestCase):
    """
    Tests function for Levenshtein metric calculation.
    """

    def setUp(self) -> None:
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

        self.misspelled = ["boyi", "streat", "coffe", "cta"]

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_fill_levenshtein_matrix_ideal(self):
        """
        Ideal scenario
        """
        correct_word = "word"
        misspelled_words = ["word", "ord", "cord", "board", "owrd", "drow", "different"]
        expected_matrixes = [
            [[0, 1, 2, 3, 4], [1, 0, 1, 2, 3], [2, 1, 0, 1, 2], [3, 2, 1, 0, 1], [4, 3, 2, 1, 0]],
            [[0, 1, 2, 3, 4], [1, 1, 1, 2, 3], [2, 2, 2, 1, 2], [3, 3, 3, 2, 1]],
            [[0, 1, 2, 3, 4], [1, 1, 2, 3, 4], [2, 2, 1, 2, 3], [3, 3, 2, 1, 2], [4, 4, 3, 2, 1]],
            [
                [0, 1, 2, 3, 4],
                [1, 1, 2, 3, 4],
                [2, 2, 1, 2, 3],
                [3, 3, 2, 2, 3],
                [4, 4, 3, 2, 3],
                [5, 5, 4, 3, 2],
            ],
            [[0, 1, 2, 3, 4], [1, 1, 1, 2, 3], [2, 1, 2, 2, 3], [3, 2, 2, 2, 3], [4, 3, 3, 3, 2]],
            [[0, 1, 2, 3, 4], [1, 1, 2, 3, 3], [2, 2, 2, 2, 3], [3, 3, 2, 3, 3], [4, 3, 3, 3, 4]],
            [
                [0, 1, 2, 3, 4],
                [1, 1, 2, 3, 3],
                [2, 2, 2, 3, 4],
                [3, 3, 3, 3, 4],
                [4, 4, 4, 4, 4],
                [5, 5, 5, 5, 5],
                [6, 6, 6, 5, 6],
                [7, 7, 7, 6, 6],
                [8, 8, 8, 7, 7],
                [9, 9, 9, 8, 8],
            ],
        ]
        for misspelled, expected_matrix in zip(misspelled_words, expected_matrixes):
            actual_matrix = fill_levenshtein_matrix(misspelled, correct_word)
            for expected_row, actual_row in zip(expected_matrix, actual_matrix):
                self.assertListEqual(expected_row, actual_row)

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_fill_levenshtein_matrix_bad_input(self):
        """
        Bad input argument scenario
        """
        bad_inputs = [[], {}, (), None, 42, 3.14]
        for bad_input in bad_inputs:
            self.assertIsNone(fill_levenshtein_matrix(bad_input, "word"))
            self.assertIsNone(fill_levenshtein_matrix("word", bad_input))

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_fill_levenshtein_matrix_value_check(self):
        """
        Check returned value
        """
        actual_matrix = fill_levenshtein_matrix("word1", "word2")
        self.assertIsInstance(actual_matrix, list)

        for row in actual_matrix:
            self.assertIsInstance(row, list)
            for value in row:
                self.assertIsInstance(value, int)

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_fill_levenshtein_matrix_empty_string(self):
        """
        Check return value for the empty string input
        """
        actual = fill_levenshtein_matrix("word", "")
        expected = [[0], [1], [2], [3], [4]]
        for expected_row, actual_row in zip(expected, actual):
            self.assertListEqual(expected_row, actual_row)

        actual = fill_levenshtein_matrix("", "word")
        expected = [[0, 1, 2, 3, 4]]
        for expected_row, actual_row in zip(expected, actual):
            self.assertListEqual(expected_row, actual_row)

        actual = fill_levenshtein_matrix("", "")
        expected = [[0]]
        for expected_row, actual_row in zip(expected, actual):
            self.assertListEqual(expected_row, actual_row)

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_fill_levenshtein_matrix_initialization_none(self):
        """
        Matrix initialization function returning None scenario
        """
        with mock.patch("lab_2_spellcheck.main.initialize_levenshtein_matrix", return_value=None):
            result = fill_levenshtein_matrix("ord", "word")

        self.assertIsNone(result)
