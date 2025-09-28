"""
Checks the second lab distance calculation function
"""

# pylint: disable=duplicate-code

import unittest

import pytest

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
        Jaccard Similarity metric scenario
        """
        expected_distances = [
            {  # for the misspelled word "boyi"
                "35": 1.0,
                "across": 0.875,
                "boy": 0.25,
                "cat": 1.0,
                "coffee": 0.857,
                "friend": 0.889,
                "kind": 0.857,
                "library": 0.571,
                "lived": 0.875,
                "loved": 0.875,
                "named": 1.0,
                "opened": 0.875,
                "shops": 0.857,
                "smart": 1.0,
                "stories": 0.75,
                "stories101": 0.8,
                "street": 1.0,
            },
            {  # for the misspelled word "streat"
                "35": 1.0,
                "across": 0.571,
                "boy": 1.0,
                "cat": 0.667,
                "coffee": 0.875,
                "friend": 0.778,
                "kind": 1.0,
                "library": 0.778,
                "lived": 0.889,
                "loved": 0.889,
                "named": 0.75,
                "opened": 0.889,
                "shops": 0.875,
                "smart": 0.333,
                "stories": 0.429,
                "stories101": 0.556,
                "street": 0.2,
            },
            {  # for the misspelled word "coffe"
                "35": 1.0,
                "across": 0.714,
                "boy": 0.833,
                "cat": 0.833,
                "coffee": 0.0,
                "friend": 0.75,
                "kind": 1.0,
                "library": 1.0,
                "lived": 0.875,
                "loved": 0.714,
                "named": 0.875,
                "opened": 0.714,
                "shops": 0.857,
                "smart": 1.0,
                "stories": 0.75,
                "stories101": 0.8,
                "street": 0.857,
            },
            {  # for the misspelled word "cta"
                "35": 1.0,
                "across": 0.667,
                "boy": 1.0,
                "cat": 0.0,
                "coffee": 0.833,
                "friend": 1.0,
                "kind": 1.0,
                "library": 0.875,
                "lived": 1.0,
                "loved": 1.0,
                "named": 0.857,
                "opened": 1.0,
                "shops": 1.0,
                "smart": 0.667,
                "stories": 0.875,
                "stories101": 0.9,
                "street": 0.833,
            },
        ]
        for misspelled_token, expected_dict in zip(self.misspelled, expected_distances):
            score_dict = calculate_distance(misspelled_token, self.vocabulary, self.methods[0])
            for token, metric_value in score_dict.items():
                self.assertAlmostEqual(metric_value, expected_dict[token], 3)

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_distance_by_frequency(self):
        """
        Frequency Similarity metric scenario
        """
        expected_values = [
            {
                "35": 1.0,
                "across": 1.0,
                "boy": 0.04,
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
                "street": 0.08,
            },
            {
                "35": 1.0,
                "across": 1.0,
                "boy": 1.0,
                "cat": 1.0,
                "coffee": 0.04,
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
                "cat": 0.16,
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
            score_dict = calculate_distance(misspelled_token, self.vocabulary, self.methods[1])
            for token, metric_value in score_dict.items():
                self.assertAlmostEqual(metric_value, expected_dict[token], 3)

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_distance_by_levenshtein(self):
        """
        Levenshtein Distance scenario
        """
        expected_values = [
            {  # for the misspelled word "boyi"
                "35": 4,
                "across": 5,
                "boy": 1,
                "cat": 4,
                "coffee": 5,
                "friend": 6,
                "kind": 4,
                "library": 6,
                "lived": 5,
                "loved": 4,
                "named": 5,
                "opened": 6,
                "shops": 4,
                "smart": 5,
                "stories": 5,
                "stories101": 8,
                "street": 6,
            },
            {  # for the misspelled word "streat"
                "35": 6,
                "across": 5,
                "boy": 6,
                "cat": 4,
                "coffee": 6,
                "friend": 5,
                "kind": 6,
                "library": 6,
                "lived": 5,
                "loved": 5,
                "named": 5,
                "opened": 6,
                "shops": 5,
                "smart": 4,
                "stories": 4,
                "stories101": 6,
                "street": 1,
            },
            {  # for the misspelled word "coffe"
                "35": 5,
                "across": 5,
                "boy": 4,
                "cat": 4,
                "coffee": 1,
                "friend": 6,
                "kind": 5,
                "library": 7,
                "lived": 5,
                "loved": 4,
                "named": 5,
                "opened": 5,
                "shops": 5,
                "smart": 5,
                "stories": 5,
                "stories101": 8,
                "street": 5,
            },
            {  # for the misspelled word "cta"
                "35": 3,
                "across": 5,
                "boy": 3,
                "cat": 2,
                "coffee": 5,
                "friend": 6,
                "kind": 4,
                "library": 6,
                "lived": 5,
                "loved": 5,
                "named": 5,
                "opened": 6,
                "shops": 5,
                "smart": 4,
                "stories": 6,
                "stories101": 9,
                "street": 5,
            },
        ]
        for misspelled_token, expected_dict in zip(self.misspelled, expected_values):
            score_dict = calculate_distance(misspelled_token, self.vocabulary, self.methods[2])
            for token, metric_value in score_dict.items():
                self.assertEqual(metric_value, expected_dict[token])

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark10
    def test_calculate_distance_by_jaro_winkler(self):
        """
        Jaro-Winkler Similarity scenario
        """
        expected_values = [
            {
                "35": 1.0,
                "across": 0.5277777777777779,
                "boy": 0.0,
                "cat": 1.0,
                "coffee": 0.5277777777777779,
                "friend": 0.5277777777777779,
                "kind": 1.0,
                "library": 0.5714285714285714,
                "lived": 1.0,
                "loved": 0.5166666666666666,
                "named": 1.0,
                "opened": 0.5277777777777779,
                "shops": 0.5166666666666666,
                "smart": 1.0,
                "stories": 0.40476190476190477,
                "stories101": 0.43333333333333335,
                "street": 1.0,
            },
            {
                "35": 1.0,
                "across": 0.5555555555555556,
                "boy": 1.0,
                "cat": 0.5,
                "coffee": 0.5555555555555556,
                "friend": 0.44444444444444453,
                "kind": 1.0,
                "library": 0.46031746031746035,
                "lived": 0.5444444444444445,
                "loved": 0.5444444444444445,
                "named": 0.5444444444444445,
                "opened": 0.5555555555555556,
                "shops": 0.44444444444444453,
                "smart": 0.16111111111111107,
                "stories": 0.05396825396825394,
                "stories101": 0.11111111111111122,
                "street": 0.0,
            },
            {
                "35": 1.0,
                "across": 0.42222222222222217,
                "boy": 0.48888888888888893,
                "cat": 0.38888888888888895,
                "coffee": 0.0,
                "friend": 0.42222222222222217,
                "kind": 1.0,
                "library": 1.0,
                "lived": 0.5333333333333334,
                "loved": 0.4,
                "named": 0.5333333333333334,
                "opened": 0.42222222222222217,
                "shops": 0.5333333333333334,
                "smart": 1.0,
                "stories": 0.4380952380952381,
                "stories101": 0.4666666666666667,
                "street": 0.5444444444444445,
            },
            {
                "35": 1.0,
                "across": 0.5,
                "boy": 1.0,
                "cat": 0.34444444444444455,
                "coffee": 0.4,
                "friend": 1.0,
                "kind": 1.0,
                "library": 0.5079365079365079,
                "lived": 1.0,
                "loved": 1.0,
                "named": 0.48888888888888893,
                "opened": 1.0,
                "shops": 1.0,
                "smart": 0.48888888888888893,
                "stories": 0.5079365079365079,
                "stories101": 0.5222222222222221,
                "street": 0.5,
            },
        ]
        for misspelled_token, expected_dict in zip(self.misspelled, expected_values):
            score_dict = calculate_distance(misspelled_token, self.vocabulary, self.methods[3])
            for token, metric_value in score_dict.items():
                self.assertAlmostEqual(metric_value, expected_dict[token], 3)
