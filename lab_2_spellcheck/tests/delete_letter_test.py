"""
Checks the second lab letter deletion function
"""

import unittest

import pytest

from lab_2_spellcheck.main import delete_letter


class DeleteLetterTest(unittest.TestCase):
    """
    Tests function for letter deletion.
    """

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_delete_letter_ideal(self):
        """
        Ideal scenario
        """
        expected = ["ord", "wod", "wor", "wrd"]
        self.assertListEqual(expected, delete_letter("word"))

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_delete_letter_length(self):
        """
        Check length of new word list
        """
        self.assertEqual(len(delete_letter("word")), 4)

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_delete_letter_bad_input(self):
        """
        Bad input scenario
        """
        bad_inputs = [[], (), {}, None, 9, 9.34, True]
        expected = []
        for bad_input in bad_inputs:
            actual = delete_letter(bad_input)
            self.assertEqual(expected, actual)

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_delete_letter_value_check(self):
        """
        Return value check
        """
        actual = delete_letter("word")
        self.assertIsInstance(actual, list)
        for word_with_deleted_letter in actual:
            self.assertIsInstance(word_with_deleted_letter, str)
