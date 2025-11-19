# pylint: disable=protected-access,duplicate-code
"""
Checks the fourth lab's model loading function.
"""
import unittest
from pathlib import Path

import pytest

from lab_4_auto_completion.main import DynamicNgramLMTrie, load


class LoadTest(unittest.TestCase):
    """
    Tests model loading functionality.
    """

    def setUp(self) -> None:
        """
        Setup for loading model
        """
        self.encoded_corpus = (
            (1, 2),
            (1, 2, 0, 3, 4, 5),
            (3, 4, 5, 0, 6, 7, 8),
            (7, 6, 8, 0, 5, 4, 8),
            (5, 4, 8, 2),
            (1, 2, 0, 5, 4, 2),
        )
        self.ngrams = 3

        self.exp_path = Path(__file__).parent.parent / "tests" / "assets" / "saved_trie.json"

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark10
    def test_load_ideal(self) -> None:
        """
        Ideal scenario for loading file
        """
        actual_model = load(str(self.exp_path))

        self.assertEqual(3, actual_model._n_gram_size)
        for text_id, tokens in enumerate(actual_model._encoded_corpus):
            self.assertTupleEqual(self.encoded_corpus[text_id], tokens)

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark10
    def test_load_return_value(self) -> None:
        """
        Checks return type for load function
        """
        self.assertIsInstance(load(str(self.exp_path)), DynamicNgramLMTrie)
