"""
Checks the second lab candidate proposal
"""

import unittest
from pathlib import Path

import pytest

from lab_2_spellcheck.main import propose_candidates


class ProposeCandidatesTest(unittest.TestCase):
    """
    Tests function for candidate proposal.
    """

    def setUp(self) -> None:
        with open(
            Path(__file__).parent / r"assets/propose_candidates_example.txt", "r", encoding="utf-8"
        ) as f:
            self.expected = f.read().splitlines()

        with open(
            Path(__file__).parent / r"assets/propose_permutations_ru.txt", "r", encoding="utf-8"
        ) as f:
            self.ru_permutations = f.read().splitlines()

        with open(
            Path(__file__).parent / r"assets/propose_permutations_en.txt", "r", encoding="utf-8"
        ) as f:
            self.en_permutations = f.read().splitlines()

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_propose_candidates_ideal(self):
        """
        Ideal scenario
        """
        actual_set = set(propose_candidates("word"))
        expected_set = set(self.expected)
        diff = expected_set.symmetric_difference(actual_set)
        self.assertSetEqual(diff, set())

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_propose_candidates_length_check(self):
        """
        Check length of candidates list
        """
        self.assertEqual(len(propose_candidates("word")), 24_254)

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_propose_candidates_value_check(self):
        """
        Return value check
        """
        candidates = propose_candidates("")
        self.assertIsInstance(candidates, tuple)
        for candidate in candidates:
            self.assertIsInstance(candidate, str)

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_propose_candidates_bad_input(self):
        """
        Bad input scenario
        """
        bad_inputs = [None, True, 42, 3.14, (), {}, []]
        for bad_input in bad_inputs:
            self.assertIsNone(propose_candidates(bad_input))

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_propose_candidates_russian(self):
        """
        Russian word scenario
        """
        actual = propose_candidates("мир")
        self.assertListEqual(sorted(actual), sorted(self.ru_permutations))

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_propose_candidates_empty_word(self):
        """
        Empty word scenario
        """
        permutations = propose_candidates("")
        self.assertListEqual(sorted(permutations), sorted(self.en_permutations))
