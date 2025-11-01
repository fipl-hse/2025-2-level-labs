"""
Checks the first lab extract significant words function
"""

import unittest

import pytest

from lab_1_keywords_tfidf.main import extract_significant_words


class ExtractSignificantWordsTest(unittest.TestCase):
    """
    Tests extract significant words function
    """

    @pytest.mark.lab_1_keywords_tfidf
    @pytest.mark.mark10
    def test_extract_significant_words_ideal(self) -> None:
        """
        Ideal extract significant words scenarios with different alpha values
        """
        chi_values = {
            "this": 0.0521,
            "is": 0.0521,
            "an": 0.0521,
            "example": 0.0521,
            "of": 0.0521,
            "test": 4.02,
            "text": 3.9,
            "contains": 0.0521,
            "two": 0.0521,
            "parts": 5.08123,
            "vale": 7.89123,
            "yarn": 15.094,
        }

        alpha = 0.05

        expected = {"parts": 5.08123, "test": 4.02, "text": 3.9, "vale": 7.89123, "yarn": 15.094}

        actual = extract_significant_words(chi_values, alpha)
        self.assertEqual(expected, actual)

        alpha = 0.01
        expected = {"vale": 7.89123, "yarn": 15.094}

        actual = extract_significant_words(chi_values, alpha)
        self.assertEqual(expected, actual)

        alpha = 0.001
        expected = {"yarn": 15.094}

        actual = extract_significant_words(chi_values, alpha)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_1_keywords_tfidf
    @pytest.mark.mark10
    def test_extract_significant_words_no_words(self) -> None:
        """
        extract significant words, no words with smaller criterion
        """
        chi_values = {"this": 0.0521, "is": 0.0521, "an": 0.0521, "parts": 0.0275}

        alpha = 0.001

        expected = {}

        actual = extract_significant_words(chi_values, alpha)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_1_keywords_tfidf
    @pytest.mark.mark10
    def test_extract_significant_words_no_chi_values(self) -> None:
        """
        extract significant words, empty chi values
        """
        chi_values = {}

        alpha = 0.001

        expected = None

        actual = extract_significant_words(chi_values, alpha)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_1_keywords_tfidf
    @pytest.mark.mark10
    def test_extract_significant_words_no_critical_chi_value(self) -> None:
        """
        extract significant words with no corresponding critical chi value for alpha
        """
        chi_values = {"this": 0.0521, "is": 0.0521, "an": 0.0521, "parts": 0.0275}

        alpha = 0.12873

        expected = None

        actual = extract_significant_words(chi_values, alpha)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_1_keywords_tfidf
    @pytest.mark.mark10
    def test_extract_significant_words_bad_input(self) -> None:
        """
        extract significant words invalid input
        """
        bad_inputs = ["string", (), None, True, [None]]
        expected = None
        for bad_input in bad_inputs:
            actual = extract_significant_words(bad_input, 0.001)
            self.assertEqual(expected, actual)

        for bad_input in bad_inputs:
            actual = extract_significant_words({"a": 2, "b": 3}, bad_input)
            self.assertEqual(expected, actual)

    @pytest.mark.lab_1_keywords_tfidf
    @pytest.mark.mark10
    def test_extract_significant_words_containing_bad_input(self) -> None:
        """
        extract significant words scenario with chi_values containing bad input
        """
        chi_values = {
            "this": 0.0521,
            "is": 0.0521,
            "an": 0.0521,
            "example": 0.0521,
            "of": 0.0521,
            "test": 0.0032,
            "text": 0.0356,
            "contains": 0.0521,
            "two": 0.0521,
            "parts": 0.0275,
            None: 0.03,
        }

        alpha = 0.001

        expected = None

        actual = extract_significant_words(chi_values, alpha)
        self.assertEqual(expected, actual)
