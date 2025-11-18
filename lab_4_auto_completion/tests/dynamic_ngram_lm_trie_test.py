"""
Checks Dynamic Trie Language Model Class.
"""

# pylint: disable=protected-access, duplicate-code

import unittest

import pytest

from lab_4_auto_completion.main import (
    DynamicNgramLMTrie,
    IncorrectCorpusError,
    IncorrectNgramError,
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
        self.big_trie.add_child(13)
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
            self.assertRaises(IncorrectCorpusError, model.build)

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
            parent=self.big_trie, node_name=13, freq=0.0911
        )

        self.assertEqual(assigned_found_child.get_name(), 13)
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
            self.assertIn(child.get_name(), (1, 13, 89, 7, 2))
            if child.get_name() == 13:
                self.assertEqual(child._children[0].get_name(), 42)
            if child.get_name() == 7:
                self.assertEqual(child.get_children()[0].get_name(), 4)
            if child.get_name() == 89:
                self.assertEqual(child._children[0].get_name(), 5)
                self.assertEqual(child._children[0]._children[0].get_name(), 15)
