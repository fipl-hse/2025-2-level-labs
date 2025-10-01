"""
Checks the second lab initialization of Levenshtein metric matrix
"""

import unittest

import pytest

from lab_2_spellcheck.main import initialize_levenshtein_matrix


class InitMatrixTest(unittest.TestCase):
    """
    Tests function for Levenshtein metric initialization.
    """

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_initialize_levenshtein_matrix_ideal(self):
        """
        Ideal scenario
        """
        candidate_len = 3
        token_len = 4
        expected_values = [[0, 1, 2, 3, 4], [1, 0, 0, 0, 0], [2, 0, 0, 0, 0], [3, 0, 0, 0, 0]]
        matrix = initialize_levenshtein_matrix(candidate_len, token_len)
        for row_expected, row_actual in zip(expected_values, matrix):
            self.assertListEqual(row_expected, row_actual)

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_initialize_levenshtein_matrix_bad_input(self):
        """
        Bad input argument scenario
        """
        bad_inputs = [[], {}, (), "", None, 3.14, -1]
        for bad_input in bad_inputs:
            self.assertIsNone(initialize_levenshtein_matrix(bad_input, 4))
            self.assertIsNone(initialize_levenshtein_matrix(4, bad_input))
            self.assertIsNone(initialize_levenshtein_matrix(bad_input, bad_input))

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_initialize_levenshtein_matrix_value_check(self):
        """
        Check returned value
        """
        actual_matrix = initialize_levenshtein_matrix(4, 4)
        self.assertIsInstance(actual_matrix, list)

        for row in actual_matrix:
            self.assertIsInstance(row, list)
            for value in row:
                self.assertIsInstance(value, int)

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_initialize_levenshtein_matrix_zero_input(self):
        """
        Zero input scenario
        """
        actual = initialize_levenshtein_matrix(0, 3)
        expected = [[0, 1, 2, 3]]
        for row_expected, row_actual in zip(expected, actual):
            self.assertListEqual(row_expected, row_actual)

        actual = initialize_levenshtein_matrix(3, 0)
        expected = [[0], [1], [2], [3]]
        self.assertListEqual(expected, actual)

        actual = initialize_levenshtein_matrix(0, 0)
        expected = [[0]]
        for row_expected, row_actual in zip(expected, actual):
            self.assertListEqual(row_expected, row_actual)
