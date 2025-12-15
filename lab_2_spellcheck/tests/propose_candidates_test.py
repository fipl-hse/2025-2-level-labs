"""
Checks the second lab candidate proposal
"""

# pylint: disable=duplicate-code

import unittest
from pathlib import Path
from unittest import mock

import pytest

from lab_2_spellcheck.main import propose_candidates


class ProposeCandidatesTest(unittest.TestCase):
    """
    Tests function for candidate proposal.
    """

    def setUp(self) -> None:
        """
        Set up function of candidate proposal tests class.
        """
        with open(
            Path(__file__).parent / r"assets/propose_candidates_example.txt", "r", encoding="utf-8"
        ) as f:
            self.expected_en = tuple(f.read().splitlines())

        with open(
            Path(__file__).parent / r"assets/propose_permutations_ru.txt", "r", encoding="utf-8"
        ) as f:
            self.permutations_ru = tuple(f.read().splitlines())

        with open(
            Path(__file__).parent / r"assets/propose_permutations_en.txt", "r", encoding="utf-8"
        ) as f:
            self.permutations_en = tuple(f.read().splitlines())

        self.alphabet_ru = list("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")
        self.alphabet_en = list("abcdefghijklmnopqrstuvwxyz")

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_propose_candidates_ideal(self) -> None:
        """
        Ideal scenario
        """
        self.assertTupleEqual(propose_candidates("word", self.alphabet_en), self.expected_en)

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_propose_candidates_length_check(self) -> None:
        """
        Check length of candidates list
        """
        self.assertEqual(len(propose_candidates("word", self.alphabet_en)), 24_254)

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_propose_candidates_value_check(self) -> None:
        """
        Return value check
        """
        candidates = propose_candidates("", self.alphabet_en)
        self.assertIsInstance(candidates, tuple)
        for candidate in candidates:
            self.assertIsInstance(candidate, str)

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_propose_candidates_bad_input(self) -> None:
        """
        Bad input scenario
        """
        bad_words = [None, True, 42, 3.14, (), {}, []]
        bad_alphabets = [None, True, 42, 3.14, (), {}, ""]
        for bad_word in bad_words:
            for bad_alphabet in bad_alphabets:
                self.assertIsNone(propose_candidates(bad_word, self.alphabet_en))
                self.assertIsNone(propose_candidates("word", bad_alphabet))
                self.assertIsNone(propose_candidates(bad_word, bad_alphabet))

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_propose_candidates_russian(self) -> None:
        """
        Russian word scenario
        """
        self.assertSetEqual(
            set(propose_candidates("мир", self.alphabet_ru)), set(self.permutations_ru)
        )

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_propose_candidates_empty_word(self) -> None:
        """
        Empty word scenario
        """
        permutations = propose_candidates("", self.alphabet_en)
        self.assertTupleEqual(permutations, self.permutations_en)

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_propose_candidates_empty_inputs(self) -> None:
        """
        Empty word scenario
        """
        self.assertTupleEqual(propose_candidates("", []), ())

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_propose_candidates_generate_candidates_none(self) -> None:
        """
        Candidates generation function returning None scenario
        """
        with mock.patch("lab_2_spellcheck.main.generate_candidates", return_value=None):
            result = propose_candidates("ord", self.alphabet_en)

        self.assertIsNone(result)

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_propose_candidates_generate_secondary_candidates_none(self) -> None:
        """
        Candidates generation function returning None for new candidates scenario
        """
        with mock.patch(
            "lab_2_spellcheck.main.generate_candidates", side_effect=[["proper", "candidate"], None]
        ):
            result = propose_candidates("", self.alphabet_en)
        self.assertIsNone(result)
