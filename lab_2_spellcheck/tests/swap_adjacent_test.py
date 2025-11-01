"""
Checks the second lab letter swapping function
"""

import unittest

import pytest

from lab_2_spellcheck.main import swap_adjacent


class SwapAdjacentTest(unittest.TestCase):
    """
    Tests function for letter swapping.
    """

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_swap_adjacent_ideal(self) -> None:
        """
        Ideal scenario
        """
        self.assertListEqual(swap_adjacent("word"), ["owrd", "wodr", "wrod"])

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_swap_adjacent_length_check(self) -> None:
        """
        Check length of new word list
        """
        self.assertEqual(len(swap_adjacent("word")), 3)

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_swap_adjacent_bad_input(self) -> None:
        """
        Bad input scenario
        """
        bad_inputs = [[], (), {}, None, 9, 9.34, True]
        expected = []
        for bad_input in bad_inputs:
            actual = swap_adjacent(bad_input)
            self.assertEqual(expected, actual)

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_swap_adjacent_value_check(self) -> None:
        """
        Return value check
        """
        actual = swap_adjacent("word")
        self.assertIsInstance(actual, list)
        for word_with_swapped_letters in actual:
            self.assertIsInstance(word_with_swapped_letters, str)
