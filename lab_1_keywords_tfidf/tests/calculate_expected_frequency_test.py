"""
Checks the first lab calculate expected frequencies function
"""

import unittest

import pytest

from lab_1_keywords_tfidf.main import calculate_expected_frequency


class CalculateExpectedFrequencyTest(unittest.TestCase):
    """
    Tests calculating expected frequency function
    """

    @pytest.mark.lab_1_keywords_tfidf
    @pytest.mark.mark10
    def test_calculate_expected_frequency_ideal(self) -> None:
        """
        Ideal calculate expected frequency scenario
        """
        tf_doc = {
            "this": 2,
            "is": 4,
            "an": 5,
            "example": 2,
            "of": 2,
            "test": 2,
            "text": 2,
            "contains": 2,
            "two": 2,
            "parts": 2,
        }

        tf_corpus = {
            "this": 10,
            "is": 30,
            "an": 25,
            "example": 5,
            "of": 12,
            "test": 3,
            "text": 5,
            "contains": 3,
            "two": 4,
            "parts": 3,
        }

        expected = {
            "an": 6.0,
            "contains": 1.0,
            "example": 1.4,
            "is": 6.8,
            "of": 2.8,
            "parts": 1.0,
            "test": 1.0,
            "text": 1.4,
            "this": 2.4,
            "two": 1.2,
        }

        actual = calculate_expected_frequency(tf_doc, tf_corpus)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_1_keywords_tfidf
    @pytest.mark.mark10
    def test_calculate_expected_frequency_no_tf_doc(self) -> None:
        """
        calculate expected frequency, empty tf_doc
        """
        tf_doc = {}

        tf_corpus = {
            "this": 10,
            "is": 30,
            "an": 25,
            "example": 5,
            "of": 12,
            "test": 3,
            "text": 5,
            "contains": 3,
            "two": 4,
            "parts": 3,
        }

        expected = None

        actual = calculate_expected_frequency(tf_doc, tf_corpus)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_1_keywords_tfidf
    @pytest.mark.mark10
    def test_calculate_expected_frequency_no_tf_corpus(self) -> None:
        """
        calculate expected frequency, empty tf_corpus
        """
        tf_doc = {
            "this": 2,
            "is": 4,
            "an": 5,
            "example": 2,
            "of": 2,
            "test": 2,
            "text": 2,
            "contains": 2,
            "two": 2,
            "parts": 2,
        }

        tf_corpus = {}

        expected = {
            "an": 5.0,
            "contains": 2.0,
            "example": 2.0,
            "is": 4.0,
            "of": 2.0,
            "parts": 2.0,
            "test": 2.0,
            "text": 2.0,
            "this": 2.0,
            "two": 2.0,
        }

        actual = calculate_expected_frequency(tf_doc, tf_corpus)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_1_keywords_tfidf
    @pytest.mark.mark10
    def test_calculate_expected_frequency_bad_input(self) -> None:
        """
        calculate expected frequency invalid input
        """
        bad_inputs = ["string", (), None, 9, 9.34, True, [None]]
        expected = None
        for bad_input in bad_inputs:
            actual = calculate_expected_frequency(bad_input, {"a": 2, "b": 3})
            self.assertEqual(expected, actual)

        for bad_input in bad_inputs:
            actual = calculate_expected_frequency({"a": 2, "b": 3}, bad_input)
            self.assertEqual(expected, actual)

    @pytest.mark.lab_1_keywords_tfidf
    @pytest.mark.mark10
    def test_calculate_expected_frequency_containing_bad_input(self) -> None:
        """
        calculate expected frequency scenario with tf_doc containing bad input
        """
        tf_doc = {
            "this": 2,
            "is": 4,
            "an": 5,
            "example": 2,
            "of": 2,
            "test": 2,
            "text": 2,
            "contains": 2,
            "two": 2,
            "parts": 2,
            None: 2,
        }

        tf_corpus = {
            "this": 10,
            "is": 30,
            "an": 25,
            "example": 5,
            "of": 12,
            "test": 3,
            "text": 5,
            "contains": 3,
            "two": 4,
            "parts": 3,
        }

        expected = None

        actual = calculate_expected_frequency(tf_doc, tf_corpus)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_1_keywords_tfidf
    @pytest.mark.mark10
    def test_calculate_expected_frequency_no_tf_for_term(self) -> None:
        """
        calculate expected frequency when no tf in corpus
        """
        tf_doc = {
            "this": 2,
            "is": 4,
            "an": 5,
            "example": 2,
            "of": 2,
            "test": 2,
            "text": 2,
            "contains": 2,
            "two": 2,
            "parts": 2,
        }

        tf_corpus = {
            "this": 10,
            "is": 30,
            "an": 25,
            "example": 5,
            "of": 12,
            "test": 3,
            "text": 5,
            "contains": 3,
            "two": 7,
        }

        expected = {
            "an": 6.0,
            "contains": 1.0,
            "example": 1.4,
            "is": 6.8,
            "of": 2.8,
            "parts": 0.4,
            "test": 1.0,
            "text": 1.4,
            "this": 2.4,
            "two": 1.8,
        }

        actual = calculate_expected_frequency(tf_doc, tf_corpus)
        self.assertEqual(expected, actual)
