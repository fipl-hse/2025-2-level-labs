"""
Checks the second lab function for counting transpositions
"""

import unittest

import pytest

from lab_2_spellcheck.main import count_transpositions


class CountTranspositionsTest(unittest.TestCase):
    """
    Tests function for counting transpositions.
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
    def test_get_count_transpositions_ideal(self):
        """
        Ideal scenario
        """
        correct_word = "word"
        misspelled_words = ["word", "ord", "cord", "board", "owrd", "drow", "different"]

        expected = [0, 0, 0, 0, 1, 2, 1]
        actual = []

        for misspelled in misspelled_words:
            _, token_matches, candidate_matches = self.matches[misspelled]
            actual.append(
                count_transpositions(misspelled, correct_word, token_matches, candidate_matches)
            )

        self.assertListEqual(expected, actual)

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark10
    def test_get_count_transpositions_value_check(self):
        """
        Check return value
        """
        match = self.matches["word"][1]
        self.assertIsInstance(count_transpositions("word", "word", match, match), int)

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark10
    def test_get_count_transpositions_bad_input(self):
        """
        Check return value
        """
        good_match = self.matches["word"][1]
        bad_matches = [None, 42, 3.14, True, "", (), [], {}]
        bad_tokens = [None, 42, 3.14, True, [], {}, ()]

        for bad_word in bad_tokens:
            self.assertIsNone(count_transpositions(bad_word, "word", good_match, good_match))
            self.assertIsNone(count_transpositions("word", bad_word, good_match, good_match))
            self.assertIsNone(count_transpositions(bad_word, bad_word, good_match, good_match))
            for bad_match in bad_matches:
                self.assertIsNone(count_transpositions("word1", "word2", bad_match, good_match))
                self.assertIsNone(count_transpositions("word1", "word2", good_match, bad_match))
                self.assertIsNone(count_transpositions(bad_word, bad_word, bad_match, bad_match))
