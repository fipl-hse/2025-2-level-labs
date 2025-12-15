"""
Checks the second lab letter addition function
"""

import unittest
from pathlib import Path

import pytest

from lab_2_spellcheck.main import add_letter


class AddLetterTest(unittest.TestCase):
    """
    Tests function for letter addition.
    """

    def setUp(self) -> None:
        """
        Set up for letter addition tests class.
        """
        self.alphabet = list("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")

        with open(
            Path(__file__).parent / r"assets/add_letter_example.txt", "r", encoding="utf-8"
        ) as f:
            self.expected = f.read().splitlines()

        self.new_words_count = 165  # 33 letters for 5 positions

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_add_letter_ideal(self) -> None:
        """
        Ideal scenario
        """
        self.assertListEqual(self.expected, add_letter("word", self.alphabet))

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_add_letter_length_check(self) -> None:
        """
        Check length of new word list
        """
        self.assertEqual(len(add_letter("word", self.alphabet)), self.new_words_count)

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_add_letter_bad_input(self) -> None:
        """
        Bad input scenario
        """
        bad_inputs = [(), {}, None, 9, 9.34, True, []]
        expected = []
        for bad_input in bad_inputs:
            actual = add_letter(bad_input, self.alphabet)
            self.assertEqual(expected, actual)

            actual = add_letter("word", bad_input)
            self.assertEqual(expected, actual)

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_add_letter_value_check(self) -> None:
        """
        Return value check
        """
        actual = add_letter("word", self.alphabet)
        self.assertIsInstance(actual, list)
        for word_with_added_letter in actual:
            self.assertIsInstance(word_with_added_letter, str)
