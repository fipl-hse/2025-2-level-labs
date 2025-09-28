"""
Checks the second lab candidate generation
"""

import unittest
from pathlib import Path

import pytest

from lab_2_spellcheck.main import generate_candidates


class GenerateCandidatesTest(unittest.TestCase):
    """
    Tests function for candidate generation.
    """

    def setUp(self) -> None:
        with open(
            Path(__file__).parent / r"assets/generate_candidates_example.txt", "r", encoding="utf-8"
        ) as f:
            self.expected = f.read().splitlines()

        self.alphabet = list("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")
        self.alphabet_en = list("abcdefghijklmnopqrstuvwxyz")

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_generate_candidates_ideal(self):
        """
        Ideal scenario
        """
        self.assertListEqual(self.expected, generate_candidates("word", self.alphabet))

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_generate_candidates_length_check(self):
        """
        Check length of candidates list
        """
        number_of_candidates = len(generate_candidates("word", self.alphabet))
        # Augmentation stats:
        # 4 candidates via letter deletion
        # 165 candidates via letter addition
        # 132 candidates via letter replacement
        # 3 candidates via letter swapping
        # Overall new candidates: 304
        expected_number_of_candidates = 304
        self.assertEqual(number_of_candidates, expected_number_of_candidates)

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_generate_candidates_value_check(self):
        """
        Return value check
        """
        candidates = generate_candidates("word", [])
        self.assertIsInstance(candidates, list)
        for candidate in candidates:
            self.assertIsInstance(candidate, str)

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_generate_candidates_bad_input(self):
        """
        Bad input scenario
        """
        bad_words = [None, True, 42, 3.14, (), {}, []]
        bad_alphabets = [None, True, 42, 3.14, (), {}, ""]
        for bad_word in bad_words:
            for bad_alphabet in bad_alphabets:
                self.assertIsNone(generate_candidates(bad_word, self.alphabet))
                self.assertIsNone(generate_candidates("word", bad_alphabet))
                self.assertIsNone(generate_candidates(bad_word, bad_alphabet))

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_generate_candidates_empty_word(self):
        """
        Empty word scenario
        """
        self.assertSetEqual(set(generate_candidates("", self.alphabet)), set(self.alphabet))

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_generate_candidates_empty_inputs(self):
        """
        Empty word scenario
        """
        self.assertListEqual(generate_candidates("", []), [])

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_generate_candidates_alphabet_en(self):
        """
        Ideal scenario
        """
        self.assertListEqual(generate_candidates("", self.alphabet_en), self.alphabet_en)
