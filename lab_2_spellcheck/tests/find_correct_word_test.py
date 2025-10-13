"""
Checks the second lab word search function
"""

import unittest
from unittest import mock

import pytest

from lab_2_spellcheck.main import find_correct_word


class FindCorrectWordTest(unittest.TestCase):
    """
    Tests function for word search.
    """

    def setUp(self) -> None:
        self.vocabulary = {
            "35": 0.04,
            "across": 0.08,
            "boy": 0.04,
            "cat": 0.16,
            "coffee": 0.04,
            "friend": 0.04,
            "kind": 0.04,
            "library": 0.12,
            "lived": 0.04,
            "loved": 0.08,
            "named": 0.04,
            "opened": 0.04,
            "shops": 0.04,
            "smart": 0.04,
            "stories": 0.04,
            "stories101": 0.04,
            "street": 0.08,
        }

        self.misspelled = ["boyi", "streat", "coffe", "cta"]
        self.expected = ["boy", "street", "coffee", "cat"]

        self.methods = ["jaccard", "frequency-based", "levenshtein", "jaro-winkler"]
        self.alphabet_en = list("abcdefghijklmnopqrstuvwxyz")

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_find_correct_word_bad_input(self):
        """
        Bad input scenario
        """
        good_token = "token"

        bad_tokens = [None, True, 42, 3.14, (), {}, []]
        bad_vocabularies = [None, True, 42, 3.14, (), "document", [], {}, {"good": "bad"}, {1: 0}]
        bad_methods = ["jacard", None, True, 42, 3.14, (), "", [], {}]
        bad_alphabets = [True, 42, 3.14, (), {}, ""]

        for bad_input in bad_tokens:
            self.assertIsNone(find_correct_word(bad_input, self.vocabulary, self.methods[0]))

        for bad_input in bad_vocabularies:
            self.assertIsNone(find_correct_word(good_token, bad_input, self.methods[0]))

        for bad_input in bad_methods:
            self.assertIsNone(find_correct_word(good_token, self.vocabulary, bad_input))

        for bad_input in bad_alphabets:
            self.assertIsNone(
                find_correct_word(good_token, self.vocabulary, self.methods[2], bad_input)
            )

        self.assertIsNone(
            find_correct_word(bad_tokens[0], bad_vocabularies[0], bad_methods[1], bad_alphabets[0])
        )

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_find_correct_word_return_check(self):
        """
        Check return value
        """
        actual = find_correct_word(self.misspelled[0], self.vocabulary, self.methods[0])
        self.assertIsInstance(actual, str)

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_find_correct_word_by_jaccard(self):
        """
        Jaccard distance metric scenario
        """
        for misspelled_token, expected_word in zip(self.misspelled, self.expected):
            self.assertEqual(
                find_correct_word(misspelled_token, self.vocabulary, self.methods[0]), expected_word
            )

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_find_correct_word_by_frequency(self):
        """
        Frequency distance metric scenario
        """
        for misspelled_token, expected_word in zip(self.misspelled, self.expected):
            self.assertEqual(
                find_correct_word(
                    misspelled_token, self.vocabulary, self.methods[1], self.alphabet_en
                ),
                expected_word,
            )

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_find_correct_word_by_levenshtein(self):
        """
        Levenshtein distance metric scenario
        """
        for misspelled_token, expected_word in zip(self.misspelled, self.expected):
            self.assertEqual(
                find_correct_word(misspelled_token, self.vocabulary, self.methods[2]), expected_word
            )

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark10
    def test_find_correct_word_by_jaro_winkler(self):
        """
        Jaro-Winkler distance metric scenario
        """
        for misspelled_token, expected_word in zip(self.misspelled, self.expected):
            self.assertEqual(
                find_correct_word(misspelled_token, self.vocabulary, self.methods[3]), expected_word
            )

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_find_correct_word_calculate_distance_none(self):
        """
        Calculate distance function returning None scenario
        """
        with mock.patch("lab_2_spellcheck.main.calculate_distance", return_value=None):
            result = find_correct_word("word", self.vocabulary, self.methods[0])

        self.assertIsNone(result)

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_find_correct_word_by_frequency_several_candidates(self):
        """
        Case of several candidates being close.
        """
        actual = find_correct_word("laved", self.vocabulary, "frequency-based", self.alphabet_en)
        # there are several non 1.0 candidates in the dictionary:
        # "lived": 0.96, "loved": 0.92, "named": 0.96
        # "loved" is the closest by distance
        self.assertEqual("loved", actual)
