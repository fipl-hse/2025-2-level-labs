"""
Checks Prefix Trie Class.
"""

# pylint: disable=protected-access

import unittest
from unittest import mock

import pytest

from lab_4_auto_completion.main import PrefixTrie, TriePrefixNotFoundError


class PrefixTrieTest(unittest.TestCase):
    """
    Tests PrefixTrie class functionality
    """

    def setUp(self) -> None:
        """
        Setup for PrefixTrieTest
        """
        self.trie = PrefixTrie()
        self.encoded_corpus = ((1, 2, 3), (1, 2, 4), (1, 3, 5), (2, 3, 4))

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_prefix_trie_initialization(self) -> None:
        """
        PrefixTrie initialization scenario.
        """
        self.assertIsNotNone(self.trie._root)
        self.assertEqual(len(self.trie._root.get_children()), 0)

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_prefix_trie_str_representation_ideal(self) -> None:
        """
        Ideal PrefixTrie __str__ scenario.
        """
        str_repr = str(self.trie)
        self.assertIn("PrefixTrie", str_repr)

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_prefix_trie_fill_ideal(self) -> None:
        """
        Ideal fill scenario.
        """
        self.trie.fill(self.encoded_corpus)
        self.assertTrue(len(self.trie._root.get_children()) > 0)

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_prefix_trie_clean_ideal(self) -> None:
        """
        Ideal clean scenario.
        """
        self.trie.fill(self.encoded_corpus)
        self.trie.clean()
        self.assertEqual(len(self.trie._root.get_children()), 0)

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_prefix_trie_get_prefix_ideal(self) -> None:
        """
        Ideal get_prefix scenario.
        """
        self.trie.fill(self.encoded_corpus)
        prefix_node = self.trie.get_prefix((1, 2))
        self.assertIsNotNone(prefix_node)
        self.assertTrue(prefix_node.has_children())

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_prefix_trie_get_prefix_not_found(self) -> None:
        """
        get_prefix with non-existent prefix scenario.
        """
        self.trie.fill(self.encoded_corpus)
        with self.assertRaises(TriePrefixNotFoundError):
            self.trie.get_prefix((99, 100))

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_prefix_trie_suggest_ideal(self) -> None:
        """
        Ideal suggest scenario.
        """
        self.trie.fill(self.encoded_corpus)
        suggestions = self.trie.suggest((1,))
        self.assertIsInstance(suggestions, tuple)
        self.assertGreater(len(suggestions), 0)

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_suggest_node_without_children(self) -> None:
        """
        Start_node has no children scenario.
        """
        simple_sequence = ((1, 2, 3),)
        self.trie.fill(simple_sequence)

        suggestions = self.trie.suggest((1, 2, 3))
        self.assertEqual(suggestions, ())

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_suggest_if_child_data_is_none(self) -> None:
        """
        Checks that suggest method skips None-data children.
        """
        self.trie.fill(self.encoded_corpus)

        with mock.patch.object(self.trie, "get_prefix") as mock_get_prefix:
            mock_node = mock.MagicMock()

            child_with_data = mock.MagicMock()
            child_with_data.get_data.return_value = 3

            child_with_none = mock.MagicMock()
            child_with_none.get_data.return_value = None

            mock_node.get_children.return_value = [child_with_data, child_with_none]
            mock_get_prefix.return_value = mock_node

            self.trie.suggest((1, 2))

            mock_node.get_children.assert_called_once()
