"""
Checks the first lab calculate chi values function
"""

import unittest

import pytest

from lab_1_keywords_tfidf.main import calculate_chi_values


class CalculateChiValuesTest(unittest.TestCase):
    """
    Tests calculating chi values function
    """

    @pytest.mark.lab_1_keywords_tfidf
    @pytest.mark.mark10
    def test_calculate_chi_values_ideal(self):
        """
        Ideal calculate chi values scenario
        """
        expected_frequencies = {"this": 0.1, "is": 0.4, "example": 0.2}

        observed_frequencies = {"this": 1, "is": 4, "example": 2}

        expected = {"example": 16.2, "is": 32.4, "this": 8.1}

        actual = calculate_chi_values(expected_frequencies, observed_frequencies)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_1_keywords_tfidf
    @pytest.mark.mark10
    def test_calculate_chi_values_no_tf_doc(self):
        """
        calculate chi values, empty expected
        """
        expected_frequencies = {}

        observed_frequencies = {"this": 1, "is": 4, "example": 2}

        expected = None

        actual = calculate_chi_values(expected_frequencies, observed_frequencies)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_1_keywords_tfidf
    @pytest.mark.mark10
    def test_calculate_chi_values_no_tf_corpus(self):
        """
        calculate chi values, empty observed
        """
        expected_frequencies = {"this": 0.1, "is": 0.4, "example": 0.2}

        observed_frequencies = {}

        expected = None

        actual = calculate_chi_values(expected_frequencies, observed_frequencies)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_1_keywords_tfidf
    @pytest.mark.mark10
    def test_calculate_chi_values_bad_input(self):
        """
        calculate chi values invalid input
        """
        bad_inputs = ["string", (), None, 9, 9.34, True, [None]]
        expected = None
        for bad_input in bad_inputs:
            actual = calculate_chi_values(bad_input, {"a": 2, "b": 3})
            self.assertEqual(expected, actual)

        for bad_input in bad_inputs:
            actual = calculate_chi_values({"a": 0.2, "b": 0.3}, bad_input)
            self.assertEqual(expected, actual)

    @pytest.mark.lab_1_keywords_tfidf
    @pytest.mark.mark10
    def test_calculate_chi_values_expected_containing_bad_input(self):
        """
        calculate chi values scenario with expected containing bad input
        """
        expected_frequencies = {"this": 0.1, "is": 0.4, "example": 0.2, None: 0.1}

        observed_frequencies = {"this": 1, "is": 4, "example": 2}

        expected = None

        actual = calculate_chi_values(expected_frequencies, observed_frequencies)
        self.assertEqual(expected, actual)

        expected_frequencies = {"this": 0.1, "is": 0.4, "example": 0.2, "abc": [0.1]}
        actual = calculate_chi_values(expected_frequencies, observed_frequencies)
        self.assertEqual(expected, actual)

        expected_frequencies = {"this": 0.1, "is": 0.4, "example": 0.2, "abc": True}
        actual = calculate_chi_values(expected_frequencies, observed_frequencies)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_1_keywords_tfidf
    @pytest.mark.mark10
    def test_calculate_chi_values_observed_containing_bad_input(self):
        """
        calculate chi values scenario with observed containing bad input
        """
        expected_frequencies = {"this": 0.1, "is": 0.4, "example": 0.2}

        observed_frequencies = {"this": 1, "is": 4, "example": 2, None: 123}

        expected = None

        actual = calculate_chi_values(expected_frequencies, observed_frequencies)
        self.assertEqual(expected, actual)

        observed_frequencies = {"this": 1, "is": 4, "example": 2, "abc": [0.2]}
        actual = calculate_chi_values(expected_frequencies, observed_frequencies)
        self.assertEqual(expected, actual)
