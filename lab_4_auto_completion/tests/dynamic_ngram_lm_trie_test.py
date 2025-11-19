"""
Checks Dynamic Trie Language Model Class.
"""

# pylint: disable=protected-access, duplicate-code

import unittest
from unittest import mock

import pytest

from lab_4_auto_completion.main import (
    DynamicNgramLMTrie,
    IncorrectNgramError,
    MergeTreesError,
    NGramTrieLanguageModel,
    TrieNode,
)


class DynamicNgramLMTrieTest(unittest.TestCase):
    """
    Tests DynamicNgramLMTrie class functionality
    """

    def setUp(self) -> None:
        """
        Setup for DynamicNgramLMTrieTest
        """
        self.encoded_corpus = (
            (1, 2),
            (1, 2, 0, 3, 4, 5),
            (3, 4, 5, 0, 6, 7, 8),
            (7, 6, 8, 0, 5, 4, 8),
            (5, 4, 8, 2),
            (1, 2, 0, 5, 4, 2),
        )

        self.model = DynamicNgramLMTrie(self.encoded_corpus, 3)

        self.big_trie = TrieNode(0)
        self.big_trie.add_child(1)
        self.big_trie.add_child(12)
        self.big_trie._children[-1].add_child(42)

        self.small_trie = TrieNode(7)
        self.small_trie.add_child(2)

        self.huge_trie = TrieNode(0)
        self.huge_trie.add_child(89)
        self.huge_trie.add_child(7)
        self.huge_trie._children[0].add_child(5)
        self.huge_trie._children[0]._children[0].add_child(15)
        self.huge_trie._children[1].add_child(4)

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark10
    def test_initialization_ideal(self) -> None:
        """
        DynamicNgramLMTrie initialization ideal scenario.
        """
        self.assertEqual(self.model._max_ngram_size, 3)
        self.assertEqual(self.model._current_n_gram_size, 0)

        self.assertIsNotNone(self.model._root)
        self.assertIsInstance(self.model._root, TrieNode)

        for text_id, tokens in enumerate(self.model._encoded_corpus):
            self.assertTupleEqual(self.encoded_corpus[text_id], tokens)

        self.assertDictEqual(self.model._models, {})

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark10
    def test_build_ideal(self) -> None:
        """
        Ideal DynamicNgramLMTrie build scenario.
        """
        result = self.model.build()
        self.assertEqual(result, 0)
        self.assertSetEqual({3, 2}, set(self.model._models.keys()))
        self.assertTrue(self.model._root.has_children())

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark10
    def test_build_invalid_input(self) -> None:
        """
        DynamicNgramLMTrie build scenario for invalid input.
        """
        bad_corpora = [42, 3.14, [], {}, (), "string"]

        for bad_corpus in bad_corpora:
            model = DynamicNgramLMTrie(bad_corpus)
            self.assertEqual(model.build(), 1)

        bad_ngram_sizes = [1, 0, -42, 3.14, [], {}, (), "string"]

        for bad_size in bad_ngram_sizes:
            model = DynamicNgramLMTrie(self.encoded_corpus, bad_size)
            self.assertEqual(model.build(), 1)

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark10
    def test_set_current_ngram_size_ideal(self) -> None:
        """
        DynamicNgramLMTrie set_current_ngram_size ideal scenario.
        """
        self.model.set_current_ngram_size(3)

        self.assertEqual(self.model._current_n_gram_size, 3)

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark10
    def test_set_current_ngram_size_invalid_input(self) -> None:
        """
        DynamicNgramLMTrie set_current_ngram_size invalid input scenario.
        """
        bad_ngram_sizes = [42, 1, 0, -1, 3.14, [], {}, (), "string"]

        for bad_size in bad_ngram_sizes:
            self.assertRaises(IncorrectNgramError, self.model.set_current_ngram_size, bad_size)

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark10
    def test_generate_next_token_ideal(self) -> None:
        """
        Ideal generate_next_token scenario.
        """
        self.model.build()
        self.model.set_current_ngram_size(3)

        expected = {0: 0.0625}

        next_tokens = self.model.generate_next_token((1, 2))
        for token, value in next_tokens.items():
            self.assertEqual(expected[token], value)

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark10
    def test_generate_next_token_invalid_input(self) -> None:
        """
        Invalid inputs for generate_next_token scenario.
        """
        self.model.build()
        self.model.set_current_ngram_size(3)

        bad_inputs = [1, [None], {}, None, (), 1.1, True, "hey"]
        for bad_input in bad_inputs:
            self.assertIsNone(self.model.generate_next_token(bad_input))

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark10
    def test_generate_next_token_short_sequence(self) -> None:
        """
        DynamicNgramLMTrie generate_next_token with short sequence scenario.
        """
        self.model.build()

        expected = {num: 0.0 for num in range(9)}

        next_tokens = self.model.generate_next_token((1,))
        for token, value in next_tokens.items():
            self.assertEqual(expected[token], value)

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark10
    def test_assign_child_ideal(self) -> None:
        """
        DynamicNgramLMTrie _assign_child ideal scenario.
        """
        assigned_found_child = self.model._assign_child(
            parent=self.big_trie, node_name=12, freq=0.0911
        )

        self.assertEqual(assigned_found_child.get_name(), 12)
        self.assertEqual(assigned_found_child.get_value(), 0.0911)
        self.assertEqual(assigned_found_child.get_children()[0].get_name(), 42)

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark10
    def test_assign_child_missing_child(self) -> None:
        """
        DynamicNgramLMTrie _assign_child no child in parent trie scenario.
        """
        assigned_found_child = self.model._assign_child(parent=self.big_trie, node_name=7)

        self.assertEqual(assigned_found_child.get_name(), 7)
        self.assertEqual(assigned_found_child.get_value(), 0.0)
        self.assertFalse(assigned_found_child.has_children())

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark10
    def test_assign_child_missing_child_no_name(self) -> None:
        """
        DynamicNgramLMTrie _assign_child no child in parent trie and no name to assign scenario.
        """
        self.assertRaises(ValueError, self.model._assign_child, self.big_trie, None)

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark10
    def test_insert_trie_ideal(self) -> None:
        """
        DynamicNgramLMTrie _insert_trie ideal scenario.
        """
        self.model._root = TrieNode()
        self.model._insert_trie(self.small_trie)

        self.assertEqual(self.model._root.get_name(), None)
        self.assertTrue(self.model._root.has_children())
        children = self.model._root.get_children()
        self.assertEqual(len(children), 1)
        self.assertEqual(children[0].get_name(), 2)

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark10
    def test_insert_trie_complex_trie(self) -> None:
        """
        DynamicNgramLMTrie _insert_trie scenario for _root being
        partially filled.
        """
        self.model._root = self.huge_trie

        new_trie = TrieNode(111)
        new_trie.add_child(89)
        child_89 = new_trie.get_children()[0]
        child_89.add_child(5)
        child_5 = child_89.get_children()[0]
        child_5.add_child(1)
        child_5.add_child(15)
        child_1 = child_5.get_children()[0]
        child_15 = child_5.get_children()[1]
        child_1.add_child(5)
        child_15.add_child(0)

        self.model._insert_trie(new_trie)

        self.assertEqual(self.model._root.get_name(), 0)
        self.assertTrue(self.model._root.has_children())

        root_children = self.model._root.get_children()
        self.assertEqual(len(root_children), 2)
        self.assertEqual(root_children[0].get_name(), 89)
        self.assertEqual(root_children[1].get_name(), 7)
        child_89 = root_children[0].get_children()[0]
        child_7 = root_children[1].get_children()[0]
        self.assertEqual(child_7.get_name(), 4)
        self.assertEqual(child_89.get_name(), 5)
        child_5 = child_89.get_children()
        self.assertEqual(len(child_5), 2)
        self.assertEqual(child_5[0].get_name(), 15)
        self.assertEqual(child_5[1].get_name(), 1)
        self.assertEqual(child_5[0].get_children()[0].get_name(), 0)
        self.assertEqual(child_5[1].get_children()[0].get_name(), 5)

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark10
    def test_insert_trie_no_child_match(self) -> None:
        """
        DynamicNgramLMTrie _insert_trie scenario for _root children
        not having a match for insert.
        """
        self.model._root = self.big_trie

        new_trie = TrieNode()
        new_trie.add_child(42)
        new_trie.get_children()[0].add_child(2)

        self.model._insert_trie(new_trie)

        self.assertEqual(self.model._root.get_name(), 0)
        self.assertTrue(self.model._root.has_children())

        children = self.model._root.get_children()
        self.assertEqual(len(children), 3)
        self.assertEqual(children[0].get_name(), 1)
        self.assertEqual(children[1].get_name(), 12)
        self.assertEqual(children[2].get_name(), 42)

        children_12 = children[1].get_children()
        self.assertEqual(len(children_12), 1)
        self.assertEqual(children_12[0].get_name(), 42)

        children_42 = children[2].get_children()
        self.assertEqual(len(children_42), 1)
        self.assertEqual(children_42[0].get_name(), 2)

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark10
    def test_insert_trie_none_name(self) -> None:
        """
        DynamicNgramLMTrie _insert_trie scenario for source child
        node not having a name.
        """
        self.model._root = TrieNode()
        with mock.patch.object(self.small_trie, "get_name", return_value=None):
            self.model._insert_trie(self.small_trie)

        self.assertEqual(self.model._root.get_name(), None)
        self.assertTrue(self.model._root.has_children())
        children = self.model._root.get_children()
        self.assertEqual(len(children), 1)
        self.assertEqual(children[0].get_name(), 2)

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark10
    def test_merge_ideal(self) -> None:
        """
        DynamicNgramLMTrie _merge ideal scenario.
        """
        for n_gram_size in range(2, 5):
            self.model._models[n_gram_size] = NGramTrieLanguageModel(None, n_gram_size)

        self.model._models[2]._root = self.small_trie
        self.model._models[3]._root = self.big_trie
        self.model._models[4]._root = self.huge_trie

        self.model._merge()

        self.assertEqual(self.model._root.get_name(), None)
        for child in self.model._root.get_children():
            self.assertIn(child.get_name(), (1, 12, 89, 7, 2))
            if child.get_name() == 12:
                self.assertEqual(child._children[0].get_name(), 42)
            if child.get_name() == 7:
                self.assertEqual(child.get_children()[0].get_name(), 4)
            if child.get_name() == 89:
                self.assertEqual(child._children[0].get_name(), 5)
                self.assertEqual(child._children[0]._children[0].get_name(), 15)

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark10
    def test_merge_invalid_input(self) -> None:
        """
        DynamicNgramLMTrie _merge invalid_input scenario.
        """
        self.assertRaises(MergeTreesError, self.model._merge)
