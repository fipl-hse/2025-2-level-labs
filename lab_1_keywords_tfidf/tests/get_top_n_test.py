"""
Checks the first lab get top words function
"""

import unittest

import pytest

from lab_1_keywords_tfidf.main import get_top_n


class GetTopNWordsTest(unittest.TestCase):
    """
    Tests get top number of words function
    """

    @pytest.mark.lab_1_keywords_tfidf
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_get_top_n_ideal(self) -> None:
        """
        Ideal get top number of words scenario
        """
        expected = ["man"]
        actual = get_top_n({"happy": 2, "man": 3}, 1)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_1_keywords_tfidf
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_get_top_n_same_frequency(self) -> None:
        """
        Get top number of words with the same frequency check
        """
        expected = ["happy", "man"]
        actual = get_top_n({"happy": 2, "man": 2}, 2)
        self.assertEqual(expected, actual)
        expected = ["happy"]
        actual = get_top_n({"happy": 2, "man": 2}, 1)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_1_keywords_tfidf
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_get_top_n_more_number(self) -> None:
        """
        Get top number of words with bigger number of words than in dictionary
        """
        expected = ["man", "happy"]
        actual = get_top_n({"happy": 2, "man": 3}, 10)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_1_keywords_tfidf
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_get_top_n_bad_inputs(self) -> None:
        """
        Get top number of words with bad argument inputs
        """
        bad_inputs = ["string", (), None, 9, 9.34, True, [None], []]
        expected = None
        for bad_input in bad_inputs:
            actual = get_top_n(bad_input, 2)
            self.assertEqual(expected, actual)

        bad_inputs = ["string", (), None, 9.34, True, [None], []]
        expected = None
        for bad_input in bad_inputs:
            actual = get_top_n({"hey": 10, "you": 20}, bad_input)
            self.assertEqual(expected, actual)

    @pytest.mark.lab_1_keywords_tfidf
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_get_top_n_empty(self) -> None:
        """
        Get top number of words with empty arguments
        """
        expected = None
        actual = get_top_n({}, 10)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_1_keywords_tfidf
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_get_top_n_incorrect_numbers(self) -> None:
        """
        Get top number of words using incorrect number of words parameter
        """
        expected = None
        actual = get_top_n({}, -1)
        self.assertEqual(expected, actual)
        actual = get_top_n({"happy": 2}, 0)
        self.assertEqual(expected, actual)
