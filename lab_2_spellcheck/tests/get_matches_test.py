"""
Checks the second lab function for getting matches
"""

import unittest

import pytest

from lab_2_spellcheck.main import get_matches


class GetMatchesTest(unittest.TestCase):
    """
    Tests function for getting matches.
    """

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark10
    def test_get_matches_distance_ideal(self) -> None:
        """
        Ideal scenario for getting matches
        """
        correct_word = "word"
        misspelled_words = ["word", "ord", "cord", "board", "owrd", "drow", "different"]
        expected_values = [
            (4, [True, True, True, True], [True, True, True, True]),
            (3, [True, True, True], [False, True, True, True]),
            (3, [False, True, True, True], [False, True, True, True]),
            (3, [False, True, False, True, True], [False, True, True, True]),
            (4, [True, True, True, True], [True, True, True, True]),
            (4, [True, True, True, True], [True, True, True, True]),
            (
                2,
                [True, False, False, False, False, True, False, False, False],
                [False, False, True, True],
            ),
        ]
        for misspelled, expected in zip(misspelled_words, expected_values):
            value = get_matches(misspelled, correct_word, 3)
            self.assertEqual(value, expected)

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark10
    def test_get_matches_small_distance(self) -> None:
        """
        Ideal scenario for getting matches when the match distance is small
        """
        correct_word = "word"
        misspelled_words = ["word", "ord", "cord", "board", "owrd", "drow", "different"]
        expected_values = [
            (4, [True, True, True, True], [True, True, True, True]),
            (3, [True, True, True], [False, True, True, True]),
            (3, [False, True, True, True], [False, True, True, True]),
            (3, [False, True, False, True, True], [False, True, True, True]),
            (4, [True, True, True, True], [True, True, True, True]),
            (2, [False, True, True, False], [False, True, True, False]),
            (
                0,
                [False, False, False, False, False, False, False, False, False],
                [False, False, False, False],
            ),
        ]
        for misspelled, expected in zip(misspelled_words, expected_values):
            value = get_matches(misspelled, correct_word, 1)
            self.assertEqual(value, expected)

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark10
    def test_get_matches_value_check(self) -> None:
        """
        Check returned values
        """
        result = get_matches("word", "word2", 2)
        self.assertIsNotNone(result)

        if result:
            self.assertIsInstance(result, tuple)
            self.assertIsInstance(result[0], int)

            for token in result[1]:
                self.assertIsInstance(token, bool)

            for token in result[2]:
                self.assertIsInstance(token, bool)

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark10
    def test_get_matches_bad_input(self) -> None:
        """
        Bad input argument scenario
        """
        bad_str_inputs = [[], {}, (), None, 42, 3.14, True]
        bad_int_inputs = [[], {}, (), "", None, 3.14, -1]

        for bad_word in bad_str_inputs:
            self.assertIsNone(get_matches(bad_word, "word", 1))
            self.assertIsNone(get_matches("word", bad_word, 1))
            self.assertIsNone(get_matches(bad_word, bad_word, 1))
            for bad_distance in bad_int_inputs:
                self.assertIsNone(get_matches("word1", "word2", bad_distance))

                self.assertIsNone(get_matches(bad_word, bad_word, bad_distance))

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark10
    def test_get_matches_match_distance_zero(self) -> None:
        """
        Zero match distance scenario
        """
        actual = get_matches("word", "wodr", 0)
        expected = (2, [True, True, False, False], [True, True, False, False])
        self.assertIsNotNone(actual)
        if actual:
            self.assertEqual(expected[0], actual[0])
            self.assertListEqual(expected[1], actual[1])
            self.assertListEqual(expected[2], actual[2])

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark10
    def test_get_matches_empty_string(self) -> None:
        """
        Check return value for the empty string input
        """
        expected = (0, [False, False, False, False], [])
        actual = get_matches("word", "", 1)
        for expected_value, actual_value in zip(expected, actual):
            self.assertEqual(expected_value, actual_value)

        expected = (0, [], [False, False, False, False])
        actual = get_matches("", "word", 1)
        for expected_value, actual_value in zip(expected, actual):
            self.assertEqual(expected_value, actual_value)

        expected = (0, [], [])
        actual = get_matches("", "", 1)
        for expected_value, actual_value in zip(expected, actual):
            self.assertEqual(expected_value, actual_value)
