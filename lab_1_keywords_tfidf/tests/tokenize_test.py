"""
Checks the first lab text tokenizing function
"""

import unittest

import pytest

from lab_1_keywords_tfidf.main import clean_and_tokenize


class TokenizeTest(unittest.TestCase):
    """
    Tests tokenize function
    """

    @pytest.mark.lab_1_keywords_tfidf
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_tokenize_ideal(self):
        """
        Ideal tokenize scenario
        """
        expected = ["the", "weather", "is", "sunny", "the", "man", "is", "happy"]
        actual = clean_and_tokenize("The weather is sunny, the man is happy.")
        self.assertEqual(expected, actual)

    @pytest.mark.lab_1_keywords_tfidf
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_tokenize_punctuation_marks(self):
        """
        Tokenize text with different punctuation marks
        """
        expected = ["the", "first", "part", "nice", "the", "second", "part", "bad"]
        actual = clean_and_tokenize("The, first part - nice; The second part: bad!")
        self.assertEqual(expected, actual)

    @pytest.mark.lab_1_keywords_tfidf
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_tokenize_dirty_text(self):
        """
        Tokenize dirty text
        """
        expected = ["the", "first", "part", "the", "second", "part"]
        actual = clean_and_tokenize("The first% part><. The sec&*ond p@art #.")
        self.assertEqual(expected, actual)

    @pytest.mark.lab_1_keywords_tfidf
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_tokenize_with_numbers(self):
        """
        Tokenize text with several parts with numbers
        """
        expected = [
            "the",
            "first",
            "part",
            "with",
            "42",
            "the",
            "second",
            "part",
            "consists",
            "of",
            "6",
            "words",
        ]
        actual = clean_and_tokenize("The first part with 42; The second part consists of 6 words.")
        self.assertEqual(expected, actual)

    @pytest.mark.lab_1_keywords_tfidf
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_tokenize_bad_input(self):
        """
        Tokenize bad input argument scenario
        """
        bad_inputs = [[], {}, (), None, 9, 9.34, True]
        expected = None
        for bad_input in bad_inputs:
            actual = clean_and_tokenize(bad_input)
            self.assertEqual(expected, actual)
