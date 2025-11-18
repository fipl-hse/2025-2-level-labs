"""
Checks Trie Node Class.
"""

# pylint: disable=protected-access

import unittest

import pytest

from lab_4_auto_completion.main import TrieNode


class TrieNodeTest(unittest.TestCase):
    """
    Tests TrieNode class functionality.
    """

    def setUp(self) -> None:
        """
        Setup for TrieNodeTest
        """
        self.node = TrieNode(5)

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_trie_node_initialization_ideal(self) -> None:
        """
        Ideal TrieNode initialization scenario.
        """
        self.assertEqual(self.node.get_name(), 5)
        self.assertEqual(len(self.node._children), 0)

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_trie_node_str_representation_ideal(self) -> None:
        """
        Ideal TrieNode __str__ scenario.
        """
        str_repr = str(self.node)
        self.assertIn("TrieNode", str_repr)
        self.assertIn("5", str_repr)

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_trie_node_bool_representation_ideal(self) -> None:
        """
        Ideal TrieNode __bool__ scenario.
        """
        self.assertFalse(bool(self.node))
        self.node.add_child(10)
        self.assertTrue(bool(self.node))

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_trie_node_add_child_ideal(self) -> None:
        """
        Ideal add_child scenario.
        """
        initial_children_count = len(self.node.get_children())
        self.node.add_child(10)
        self.assertEqual(len(self.node.get_children()), initial_children_count + 1)

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_trie_node_has_children_ideal(self) -> None:
        """
        Ideal has_children scenario.
        """
        self.assertFalse(self.node.has_children())
        self.node.add_child(10)
        self.assertTrue(self.node.has_children())

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_trie_node_get_children_ideal(self) -> None:
        """
        Ideal get_children scenario.
        """
        self.node.add_child(10)
        self.node.add_child(20)
        children = self.node.get_children()
        self.assertEqual(len(children), 2)
        self.assertTrue(all(isinstance(child, TrieNode) for child in children))

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_trie_node_get_data_ideal(self) -> None:
        """
        Ideal get_data method scenario.
        """
        self.assertEqual(self.node.get_name(), 5)
        root_node = TrieNode()
        self.assertIsNone(root_node.get_name())
