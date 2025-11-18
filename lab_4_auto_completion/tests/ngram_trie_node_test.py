"""
Checks NGram Trie Node Class.
"""

# pylint: disable=protected-access

import unittest

import pytest

from lab_4_auto_completion.main import NGramTrieNode


class NGramTrieNodeTest(unittest.TestCase):
    """
    Tests NGramTrieNode class functionality.
    """

    def setUp(self) -> None:
        """
        Setup for NGramTrieNodeTest.
        """
        self.node = NGramTrieNode(data=5, frequency=0.5)
        self.empty_node = NGramTrieNode()

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_initialization_ideal(self) -> None:
        """
        Ideal initialization scenario.
        """
        self.assertEqual(self.node.get_data(), 5)
        self.assertEqual(self.node.get_frequency(), 0.5)

        self.assertEqual(self.empty_node.get_data(), None)
        self.assertEqual(self.empty_node.get_frequency(), 0.0)

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.skip(reason="rework")
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_str_ideal(self) -> None:
        """
        NGramTrieNode __str__ ideal.
        """
        str_representation = str(self.node)
        self.assertIn("NGramTrieNode", str_representation)
        self.assertIn("data=5", str_representation)
        self.assertIn("frequency=0.5", str_representation)

        str_representation = str(self.empty_node)
        self.assertIn("NGramTrieNode", str_representation)
        self.assertIn("data=None", str_representation)

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_add_child_ideal(self) -> None:
        """
        Ideal NGramTrieNode add_child scenario.
        """
        self.empty_node.add_child(10)

        children = self.empty_node.get_children()
        self.assertEqual(len(children), 1)
        self.assertIsInstance(children[0], NGramTrieNode)
        self.assertEqual(children[0].get_data(), 10)

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_get_set_frequency_ideal(self) -> None:
        """
        Checks NGramTrieNode frequency getter and setter.
        """
        self.assertEqual(self.node.get_frequency(), 0.5)

        self.node.set_frequency(0.88)
        self.assertEqual(self.node.get_frequency(), 0.88)

        self.empty_node.set_frequency(0.31)
        self.assertEqual(self.empty_node.get_frequency(), 0.31)

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_has_children_ideal(self) -> None:
        """
        Ideal NGramTrieNode has_children scenario.
        """
        self.assertFalse(self.node.has_children())

        self.node.add_child(15)
        self.assertTrue(self.node.has_children())

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_bool_method(self) -> None:
        """
        Checks NGramTrieNode __bool__ method.
        """
        self.assertFalse(bool(self.empty_node))

        self.empty_node.add_child(5)
        self.assertTrue(bool(self.empty_node))
