# pylint: disable=protected-access
"""
Checks the third lab's NGramLanguageModelReader class
"""
import unittest
from pathlib import Path

import pytest

from lab_3_generate_by_ngrams.main import NGramLanguageModelReader


class NGramLanguageModelReaderTest(unittest.TestCase):
    """
    Tests NGramLanguageModelReader class functionality
    """

    def setUp(self) -> None:
        """
        Setup for NGramLanguageModelReaderTest
        """
        self.path_to_test_directory = Path(__file__).parent.parent

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark10
    def test_ngram_language_model_reader_fields(self) -> None:
        """
        Checks if NGramLanguageModelReader fields are created correctly
        """
        reader = NGramLanguageModelReader(
            str(self.path_to_test_directory / "assets" / "en_for_test.json"), eow_token="_"
        )
        self.assertEqual(
            str(self.path_to_test_directory / "assets" / "en_for_test.json"), reader._json_path
        )
        self.assertEqual("_", reader._eow_token)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark10
    def test_ngram_language_model_reader_load(self) -> None:
        """
        Checks NGramLanguageModelReader load method ideal scenario
        """
        reader = NGramLanguageModelReader(
            str(self.path_to_test_directory / "assets" / "en_for_test.json"), eow_token="_"
        )
        actual = reader.load(2)
        expected_ngram_frequencies = {(0, 0): 1.0}
        self.assertEqual(2, actual.get_n_gram_size())
        for key, value in expected_ngram_frequencies.items():
            self.assertEqual(expected_ngram_frequencies.keys(), actual._n_gram_frequencies.keys())
            self.assertAlmostEqual(value, actual._n_gram_frequencies[key], places=4)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark10
    def test_ngram_language_model_reader_load_invalid_input(self) -> None:
        """
        Checks NGramLanguageModelReader load method with invalid input
        """
        reader = NGramLanguageModelReader(
            str(self.path_to_test_directory / "assets" / "en_for_test.json"), eow_token="_"
        )
        expected = None
        bad_inputs = [[None], {}, None, (), 1.1, "string"]
        for bad_input in bad_inputs:
            actual = reader.load(bad_input)
            self.assertEqual(expected, actual)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark10
    def test_ngram_language_model_reader_load_unigram(self) -> None:
        """
        Checks NGramLanguageModelReader load method with unigrams
        """
        reader = NGramLanguageModelReader(
            str(self.path_to_test_directory / "assets" / "en_for_test.json"), eow_token="_"
        )
        actual = reader.load(1)
        expected = None
        self.assertEqual(expected, actual)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark10
    def test_ngram_language_model_reader_get_text_processor(self) -> None:
        """
        Checks NGramLanguageModelReader get_text_processor method ideal scenario
        """
        reader = NGramLanguageModelReader(
            str(self.path_to_test_directory / "assets" / "en_for_test.json"), eow_token="_"
        )
        actual = reader.get_text_processor()
        expected = {
            "_": 0,
            "a": 3,
            "b": 10,
            "c": 5,
            "d": 8,
            "e": 2,
            "f": 9,
            "g": 4,
            "h": 15,
            "i": 16,
            "j": 1,
            "k": 17,
            "l": 11,
            "m": 12,
            "n": 13,
            "o": 14,
            "p": 23,
            "q": 22,
            "r": 6,
            "s": 7,
            "t": 19,
            "u": 18,
            "v": 21,
            "w": 20,
            "x": 25,
            "y": 24,
            "z": 26,
            "á": 29,
            "ä": 28,
            "é": 27,
            "ö": 31,
            "ü": 30,
        }
        self.assertEqual(expected, actual._storage)
