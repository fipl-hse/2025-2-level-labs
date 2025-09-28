"""
Checks the second lab letter replacement function
"""

import unittest
from pathlib import Path

import pytest

from lab_2_spellcheck.main import replace_letter


class ReplaceLetterTest(unittest.TestCase):
    """
    Tests function for letter replacement.
    """

    def setUp(self):
        self.alphabet = list("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")

        with open(
            Path(__file__).parent / r"assets/replace_letter_example.txt", "r", encoding="utf-8"
        ) as f:
            self.expected = f.read().splitlines()

        self.new_words_count = 132  # 33 letters in place of 4 letters

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_replace_letter_ideal(self):
        """
        Ideal scenario
        """
        self.assertListEqual(self.expected, replace_letter("word", self.alphabet))

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_replace_letter_length(self):
        """
        Check length of new word list
        """
        self.assertEqual(len(replace_letter("word", self.alphabet)), self.new_words_count)

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_replace_letter_value_check(self):
        """
        Return value check
        """
        actual = replace_letter("word", self.alphabet)
        self.assertIsInstance(actual, list)
        for word_with_replaced_letter in actual:
            self.assertIsInstance(word_with_replaced_letter, str)
