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

        self.candidates_number = 304

        self.alphabet = list("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_generate_candidates_ideal(self):
        """
        Ideal scenario
        """
        self.assertListEqual(sorted(self.expected), sorted(generate_candidates("word")))

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_generate_candidates_length_check(self):
        """
        Check length of candidates list
        """
        self.assertEqual(self.candidates_number, len(generate_candidates("word")))

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_generate_candidates_value_check(self):
        """
        Return value check
        """
        candidates = generate_candidates("word")
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
        bad_inputs = [None, True, 42, 3.14, (), {}, []]
        for bad_input in bad_inputs:
            self.assertIsNone(generate_candidates(bad_input))
            self.assertIsNone(generate_candidates("word", bad_input))
            self.assertIsNone(generate_candidates(bad_input, bad_input))

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_generate_candidates_empty_word(self):
        """
        Empty word scenario
        """
        self.assertSetEqual(set(generate_candidates("")), set(self.alphabet))

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_generate_candidates_language_en(self):
        """
        Ideal scenario
        """
        alphabet_en = "abcdefghijklmnopqrstuvwxyz"
        self.assertSetEqual(set(generate_candidates("", "en")), set(alphabet_en))
