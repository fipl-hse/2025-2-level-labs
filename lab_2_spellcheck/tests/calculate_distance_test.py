"""
Checks the second lab distance calculation function
"""

# pylint: disable=duplicate-code

import unittest
from unittest import mock

import pytest

from config.constants import FLOAT_TOLERANCE
from lab_2_spellcheck.main import calculate_distance


class CalculateDistanceTest(unittest.TestCase):
    """
    Tests function for distance calculation.
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

        self.methods = ["jaccard", "frequency-based", "levenshtein", "jaro-winkler"]

        self.alphabet = list("abcdefghijklmnopqrstuvwxyz")

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_distance_bad_input(self):
        """
        Bad input scenario
        """
        good_token = "token"

        bad_tokens = [None, True, 42, 3.14, (), {}, []]
        bad_vocabularies = [None, True, 42, 3.14, (), "document", [], {}, {"good": "bad"}, {1: 0}]
        bad_methods = ["jacard", None, True, 42, 3.14, (), "", [], {}]

        for bad_input in bad_tokens:
            self.assertIsNone(calculate_distance(bad_input, self.vocabulary, self.methods[0]))

        for bad_input in bad_vocabularies:
            self.assertIsNone(calculate_distance(good_token, bad_input, self.methods[0]))

        for bad_input in bad_methods:
            self.assertIsNone(calculate_distance(good_token, self.vocabulary, bad_input))

        self.assertIsNone(calculate_distance(bad_tokens[0], bad_vocabularies[0], bad_methods[1]))

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_distance_return_check(self):
        """
        Check return value
        """
        actual = calculate_distance(self.misspelled[0], self.vocabulary, self.methods[0])
        self.assertIsInstance(actual, dict)

        for key, value in actual.items():
            self.assertIsInstance(key, str)
            self.assertIsInstance(value, float)

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_distance_by_jaccard(self):
        """
        Jaccard distance metric scenario
        """
        expected_distances = [
            {  # for the misspelled word "boyi"
                "35": 1.0,
                "across": 0.875,
                "boy": 0.25,
                "cat": 1.0,
                "coffee": 0.8571,
                "friend": 0.8889,
                "kind": 0.8571,
                "library": 0.5714,
                "lived": 0.875,
                "loved": 0.875,
                "named": 1.0,
                "opened": 0.875,
                "shops": 0.8571,
                "smart": 1.0,
                "stories": 0.75,
                "stories101": 0.8,
                "street": 1.0,
            },
            {  # for the misspelled word "streat"
                "35": 1.0,
                "across": 0.5714,
                "boy": 1.0,
                "cat": 0.6667,
                "coffee": 0.875,
                "friend": 0.7778,
                "kind": 1.0,
                "library": 0.7778,
                "lived": 0.8889,
                "loved": 0.8889,
                "named": 0.75,
                "opened": 0.8889,
                "shops": 0.875,
                "smart": 0.3333,
                "stories": 0.4286,
                "stories101": 0.5556,
                "street": 0.2,
            },
            {  # for the misspelled word "coffe"
                "35": 1.0,
                "across": 0.7143,
                "boy": 0.8333,
                "cat": 0.8333,
                "coffee": 0.0,
                "friend": 0.75,
                "kind": 1.0,
                "library": 1.0,
                "lived": 0.875,
                "loved": 0.7143,
                "named": 0.875,
                "opened": 0.7143,
                "shops": 0.8571,
                "smart": 1.0,
                "stories": 0.75,
                "stories101": 0.8,
                "street": 0.8571,
            },
            {  # for the misspelled word "cta"
                "35": 1.0,
                "across": 0.6667,
                "boy": 1.0,
                "cat": 0.0,
                "coffee": 0.8333,
                "friend": 1.0,
                "kind": 1.0,
                "library": 0.875,
                "lived": 1.0,
                "loved": 1.0,
                "named": 0.8571,
                "opened": 1.0,
                "shops": 1.0,
                "smart": 0.6667,
                "stories": 0.875,
                "stories101": 0.9,
                "street": 0.8333,
            },
        ]
        for misspelled_token, expected_dict in zip(self.misspelled, expected_distances):
            score_dict = calculate_distance(misspelled_token, self.vocabulary, self.methods[0])
            for token, metric_value in score_dict.items():
                self.assertAlmostEqual(metric_value, expected_dict[token], FLOAT_TOLERANCE)

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_distance_by_frequency(self):
        """
        Frequency distance metric scenario
        """
        expected_values = [
            {
                "35": 1.0,
                "across": 1.0,
                "boy": 0.96,
                "cat": 1.0,
                "coffee": 1.0,
                "friend": 1.0,
                "kind": 1.0,
                "library": 1.0,
                "lived": 1.0,
                "loved": 1.0,
                "named": 1.0,
                "opened": 1.0,
                "shops": 1.0,
                "smart": 1.0,
                "stories": 1.0,
                "stories101": 1.0,
                "street": 1.0,
            },
            {
                "35": 1.0,
                "across": 1.0,
                "boy": 1.0,
                "cat": 1.0,
                "coffee": 1.0,
                "friend": 1.0,
                "kind": 1.0,
                "library": 1.0,
                "lived": 1.0,
                "loved": 1.0,
                "named": 1.0,
                "opened": 1.0,
                "shops": 1.0,
                "smart": 1.0,
                "stories": 1.0,
                "stories101": 1.0,
                "street": 0.92,
            },
            {
                "35": 1.0,
                "across": 1.0,
                "boy": 1.0,
                "cat": 1.0,
                "coffee": 0.96,
                "friend": 1.0,
                "kind": 1.0,
                "library": 1.0,
                "lived": 1.0,
                "loved": 1.0,
                "named": 1.0,
                "opened": 1.0,
                "shops": 1.0,
                "smart": 1.0,
                "stories": 1.0,
                "stories101": 1.0,
                "street": 1.0,
            },
            {
                "35": 1.0,
                "across": 1.0,
                "boy": 1.0,
                "cat": 0.84,
                "coffee": 1.0,
                "friend": 1.0,
                "kind": 1.0,
                "library": 1.0,
                "lived": 1.0,
                "loved": 1.0,
                "named": 1.0,
                "opened": 1.0,
                "shops": 1.0,
                "smart": 1.0,
                "stories": 1.0,
                "stories101": 1.0,
                "street": 1.0,
            },
        ]

        for misspelled_token, expected_dict in zip(self.misspelled, expected_values):
            score_dict = calculate_distance(
                misspelled_token, self.vocabulary, self.methods[1], self.alphabet
            )
            for token, metric_value in score_dict.items():
                self.assertAlmostEqual(metric_value, expected_dict[token], FLOAT_TOLERANCE)

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_distance_by_levenshtein(self):
        """
        Levenshtein distance scenario
        """
        expected_values = [
            {  # for the misspelled word "boyi"
                "35": 4.0,
                "across": 5.0,
                "boy": 1.0,
                "cat": 4.0,
                "coffee": 5.0,
                "friend": 6.0,
                "kind": 4.0,
                "library": 6.0,
                "lived": 5.0,
                "loved": 4.0,
                "named": 5.0,
                "opened": 6.0,
                "shops": 4.0,
                "smart": 5.0,
                "stories": 5.0,
                "stories101": 8.0,
                "street": 6.0,
            },
            {  # for the misspelled word "streat"
                "35": 6.0,
                "across": 5.0,
                "boy": 6.0,
                "cat": 4.0,
                "coffee": 6.0,
                "friend": 5.0,
                "kind": 6.0,
                "library": 6.0,
                "lived": 5.0,
                "loved": 5.0,
                "named": 5.0,
                "opened": 6.0,
                "shops": 5.0,
                "smart": 4.0,
                "stories": 4.0,
                "stories101": 6.0,
                "street": 1.0,
            },
            {  # for the misspelled word "coffe"
                "35": 5.0,
                "across": 5.0,
                "boy": 4.0,
                "cat": 4.0,
                "coffee": 1.0,
                "friend": 6.0,
                "kind": 5.0,
                "library": 7.0,
                "lived": 5.0,
                "loved": 4.0,
                "named": 5.0,
                "opened": 5.0,
                "shops": 5.0,
                "smart": 5.0,
                "stories": 5.0,
                "stories101": 8.0,
                "street": 5.0,
            },
            {  # for the misspelled word "cta"
                "35": 3.0,
                "across": 5.0,
                "boy": 3.0,
                "cat": 2.0,
                "coffee": 5.0,
                "friend": 6.0,
                "kind": 4.0,
                "library": 6.0,
                "lived": 5.0,
                "loved": 5.0,
                "named": 5.0,
                "opened": 6.0,
                "shops": 5.0,
                "smart": 4.0,
                "stories": 6.0,
                "stories101": 9.0,
                "street": 5.0,
            },
        ]
        for misspelled_token, expected_dict in zip(self.misspelled, expected_values):
            score_dict = calculate_distance(misspelled_token, self.vocabulary, self.methods[2])
            for token, metric_value in score_dict.items():
                self.assertAlmostEqual(metric_value, expected_dict[token], FLOAT_TOLERANCE)

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark10
    def test_calculate_distance_by_jaro_winkler(self):
        """
        Jaro-Winkler distance scenario
        """
        expected_values = [
            {  # for the misspelled word "boyi"
                "35": 1.0,
                "across": 0.5278,
                "boy": 0.0583,
                "cat": 1.0,
                "coffee": 0.5278,
                "friend": 0.5278,
                "kind": 1.0,
                "library": 0.5714,
                "lived": 1.0,
                "loved": 0.5167,
                "named": 1.0,
                "opened": 0.5278,
                "shops": 0.5167,
                "smart": 1.0,
                "stories": 0.4048,
                "stories101": 0.4333,
                "street": 1.0,
            },
            {  # for the misspelled word "streat"
                "35": 1.0,
                "across": 0.5556,
                "boy": 1.0,
                "cat": 0.5,
                "coffee": 0.5556,
                "friend": 0.4444,
                "kind": 1.0,
                "library": 0.4603,
                "lived": 0.5444,
                "loved": 0.5444,
                "named": 0.5444,
                "opened": 0.5556,
                "shops": 0.49,
                "smart": 0.235,
                "stories": 0.2032,
                "stories101": 0.2489,
                "street": 0.0667,
            },
            {  # for the misspelled word "coffe"
                "35": 1.0,
                "across": 0.4222,
                "boy": 0.4889,
                "cat": 0.44,
                "coffee": 0.0333,
                "friend": 0.4222,
                "kind": 1.0,
                "library": 1.0,
                "lived": 0.5333,
                "loved": 0.4,
                "named": 0.5333,
                "opened": 0.4222,
                "shops": 0.5333,
                "smart": 1.0,
                "stories": 0.4381,
                "stories101": 0.4667,
                "street": 0.5444,
            },
            {  # for the misspelled word "cta"
                "35": 1.0,
                "across": 0.5,
                "boy": 1.0,
                "cat": 0.4,
                "coffee": 0.45,
                "friend": 1.0,
                "kind": 1.0,
                "library": 0.5079,
                "lived": 1.0,
                "loved": 1.0,
                "named": 0.4889,
                "opened": 1.0,
                "shops": 1.0,
                "smart": 0.4889,
                "stories": 0.5079,
                "stories101": 0.5222,
                "street": 0.5,
            },
        ]
        for misspelled_token, expected_dict in zip(self.misspelled, expected_values):
            score_dict = calculate_distance(misspelled_token, self.vocabulary, self.methods[3])
            for token, metric_value in score_dict.items():
                self.assertAlmostEqual(metric_value, expected_dict[token], FLOAT_TOLERANCE)

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_distance_jaccard_none(self):
        """
        Jaccard distance metric returning None scenario
        """
        with mock.patch("lab_2_spellcheck.main.calculate_jaccard_distance", return_value=None):
            result = calculate_distance("boi", self.vocabulary, self.methods[0])

        self.assertIsNone(result)

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_distance_frequency_none(self):
        """
        Frequency distance metric returning None scenario
        """
        with mock.patch("lab_2_spellcheck.main.calculate_frequency_distance", return_value=None):
            result = calculate_distance("boi", self.vocabulary, self.methods[1], self.alphabet)

        self.assertIsNone(result)

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_distance_levenshtein_none(self):
        """
        Levenshtein distance returning None scenario
        """
        with mock.patch("lab_2_spellcheck.main.calculate_levenshtein_distance", return_value=None):
            result = calculate_distance("boi", self.vocabulary, self.methods[2])

        self.assertIsNone(result)

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark10
    def test_calculate_distance_jaro_winkler_none(self):
        """
        Jaro-Winkler distance returning None scenario
        """
        with mock.patch("lab_2_spellcheck.main.calculate_jaro_winkler_distance", return_value=None):
            result = calculate_distance("boi", self.vocabulary, self.methods[3])

        self.assertIsNone(result)

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_distance_frequency_no_alphabet(self):
        """
        Frequency distance metric scenario with no alphabet passed
        """
        expected = {token: 1.0 for token in self.vocabulary}
        result = calculate_distance("word", self.vocabulary, self.methods[1])

        self.assertDictEqual(result, expected)
        for token, value in result.items():
            self.assertAlmostEqual(expected[token], value)

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_distance_frequency_several_candidates(self):
        """
        Case of several candidates being close
        """
        expected = {
            "35": 1.0,
            "across": 1.0,
            "boy": 1.0,
            "cat": 1.0,
            "coffee": 1.0,
            "friend": 1.0,
            "kind": 1.0,
            "library": 1.0,
            "lived": 0.96,
            "loved": 0.92,
            "named": 0.96,
            "opened": 1.0,
            "shops": 1.0,
            "smart": 1.0,
            "stories": 1.0,
            "stories101": 1.0,
            "street": 1.0,
        }
        actual = calculate_distance("laved", self.vocabulary, self.methods[1], self.alphabet)
        for token, freq in actual.items():
            self.assertAlmostEqual(expected[token], freq, FLOAT_TOLERANCE)
