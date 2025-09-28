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
    def test_swap_adjacent_ideal(self):
        """
        Ideal scenario
        """
        self.assertListEqual(swap_adjacent("word"), ["owrd", "wodr", "wrod"])

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_swap_adjacent_length_check(self):
        """
        Check length of new word list
        """
        self.assertEqual(len(swap_adjacent("word")), 3)

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_swap_adjacent_value_check(self):
        """
        Return value check
        """
        actual = swap_adjacent("word")
        self.assertIsInstance(actual, list)
        for word_with_swapped_letters in actual:
            self.assertIsInstance(word_with_swapped_letters, str)
