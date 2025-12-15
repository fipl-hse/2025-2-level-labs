"""
Checks the second lab's vocabulary building function
"""

# pylint: disable=duplicate-code

import unittest

import pytest

from config.constants import FLOAT_TOLERANCE
from lab_2_spellcheck.main import build_vocabulary


class BuildVocabularyTest(unittest.TestCase):
    """
    Tests vocabulary building
    """

    def setUp(self) -> None:
        """
        Set up for vocabulary build tests class.
        """
        self.documents = {
            "texts": [
                "There was a boy. He was smart and kind. He had a cat.",
                "There are 35 coffee shops across the street.",
                'My friend opened a library named "Stories101"!',
                "I loved the cat that lived in the library across the street.",
                "Library cat loved stories and the other cat.",
            ],
            "clean_tokens": [
                "boy",
                "smart",
                "kind",
                "cat",
                "35",
                "coffee",
                "shops",
                "across",
                "street",
                "friend",
                "opened",
                "library",
                "named",
                "stories101",
                "loved",
                "cat",
                "lived",
                "library",
                "across",
                "street",
                "library",
                "cat",
                "loved",
                "stories",
                "cat",
            ],
        }

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_build_vocabulary_ideal(self) -> None:
        """
        Ideal scenario
        """
        expected = {
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
        actual = build_vocabulary(self.documents["clean_tokens"])

        self.assertDictEqual(actual, expected)
        for token, freq in actual.items():
            self.assertAlmostEqual(freq, expected[token], FLOAT_TOLERANCE)

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_build_vocabulary_bad_input(self) -> None:
        """
        Bad input scenario
        """
        expected = None
        bad_inputs = [None, True, 42, 3.14, (), "document", {}, [], [False, 5, {}, "bad"]]

        for bad_input in bad_inputs:
            actual = build_vocabulary(bad_input)
            self.assertEqual(expected, actual)

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_build_vocabulary_return_value(self) -> None:
        """
        Function return value check
        """
        actual = build_vocabulary(self.documents["clean_tokens"])
        self.assertIsInstance(actual, dict)
        for token, freq in actual.items():
            self.assertIsInstance(token, str)
            self.assertIsInstance(freq, float)
