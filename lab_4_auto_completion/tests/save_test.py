# pylint: disable=protected-access
"""
Checks the fourth lab's model saving function.
"""
import json
import unittest
from pathlib import Path

import pytest

from lab_4_auto_completion.main import DynamicNgramLMTrie, save


class SaveTest(unittest.TestCase):
    """
    Tests model saving functionality.
    """

    def setUp(self) -> None:
        """
        Setup for saving model
        """
        self.encoded_corpus = (
            (1, 2),
            (1, 2, 0, 3, 4, 5),
            (3, 4, 5, 0, 6, 7, 8),
            (7, 6, 8, 0, 5, 4, 8),
            (5, 4, 8, 2),
            (1, 2, 0, 5, 4, 2),
        )
        self.model = DynamicNgramLMTrie(self.encoded_corpus)
        self.model.build()

        self.exp_path = Path(__file__).parent.parent / "tests" / "assets" / "saved_trie.json"
        self.test_path = Path(__file__).parent.parent / "test_tmp" / "saved_trie.json"
        self.test_path.parent.mkdir(parents=True, exist_ok=True)

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark10
    def test_save_ideal(self) -> None:
        """
        Ideal scenario for save function
        """
        save(self.model, str(self.test_path))

        with open(str(self.test_path), "r", encoding="utf-8") as f:
            actual = json.load(f)

        with open(str(self.exp_path), "r", encoding="utf-8") as f:
            expected = json.load(f)

        self.assertEqual(actual, expected)
