# pylint: disable=duplicate-code
"""
Checks the second lab's document quantity functions
"""

import unittest

import pytest

from lab_2_spellcheck.main import find_out_of_vocab_words


class FindOutOfVocabWordsTest(unittest.TestCase):
    """
    Tests document quantity function
    """

    def setUp(self) -> None:
        self.example = {
            "text": "Kind boyi across the streat loved coffe and his cta!",
            "clean_tokens": ["kind", "boyi", "across", "streat", "loved", "coffe", "cta"],
        }
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

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_find_out_of_vocab_words_ideal(self):
        """
        Ideal scenario
        """
        actual = find_out_of_vocab_words(self.example["clean_tokens"], self.vocabulary)
        expected = ["boyi", "streat", "coffe", "cta"]
        self.assertListEqual(expected, actual)

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_find_out_of_vocab_words_bad_input(self):
        """
        Bad input scenario
        """
        good_tokens = ["proper", "list"]
        good_vocabulary = {"proper": 0.5, "list": 0.5}

        bad_tokens = [None, True, 42, 3.14, (), "document", {}, [], [False, 5, {}, "bad"]]
        bad_vocabularies = [None, True, 42, 3.14, (), "document", [], {}, {"good": "bad"}, {1: 0}]

        for bad_token in bad_tokens:
            self.assertIsNone(find_out_of_vocab_words(bad_token, good_vocabulary))
            for bad_vocabulary in bad_vocabularies:
                self.assertIsNone(find_out_of_vocab_words(good_tokens, bad_vocabulary))
                self.assertIsNone(find_out_of_vocab_words(bad_token, bad_vocabulary))

    @pytest.mark.lab_2_spellcheck
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_find_out_of_vocab_words_return_value(self):
        """
        Function return value check
        """
        actual = find_out_of_vocab_words(self.example["clean_tokens"], self.vocabulary)
        self.assertIsInstance(actual, list)
        for token in actual:
            self.assertIsInstance(token, str)
