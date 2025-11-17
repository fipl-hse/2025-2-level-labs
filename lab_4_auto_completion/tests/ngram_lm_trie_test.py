"""
Checks NGram Trie Language Model Class.
"""

# pylint: disable=protected-access

import unittest
from unittest import mock

import pytest

from lab_4_auto_completion.main import NGramTrieLanguageModel, TriePrefixNotFoundError


class NGramLMTrieTest(unittest.TestCase):
    """
    Tests NGramTrieLanguageModel class functionality
    """

    def setUp(self) -> None:
        """
        Setup for NGramTrieLanguageModelTest
        """
        self.encoded_corpus = ((1, 2, 3, 4), (1, 2, 5, 6), (2, 3, 4, 7))
        self.model = NGramTrieLanguageModel(self.encoded_corpus, 3)

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_initialization_ideal(self) -> None:
        """
        NGramTrieLanguageModel initialization ideal scenario.
        """
        self.assertEqual(self.model._n_gram_size, 3)
        self.assertEqual(self.model._encoded_corpus, self.encoded_corpus)
        self.assertIsNotNone(self.model._root)

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_str_representation_ideal(self) -> None:
        """
        NGramTrieLanguageModel __str__ ideal scenario.
        """
        str_repr = str(self.model)
        self.assertIn("NGramTrieLanguageModel", str_repr)
        self.assertIn("3", str_repr)

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_build_ideal(self) -> None:
        """
        Ideal NGramTrieLanguageModel build scenario.
        """
        result = self.model.build()
        self.assertEqual(result, 0)
        self.assertTrue(len(self.model._root.get_children()) > 0)

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_get_next_tokens_ideal(self) -> None:
        """
        get_next_tokens ideal scenario.
        """
        self.model.build()
        next_tokens = self.model.get_next_tokens((1, 2))
        self.assertIsInstance(next_tokens, dict)

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_get_next_tokens_not_found(self) -> None:
        """
        get_next_tokens with non-existent prefix scenario.
        """
        self.model.build()
        with self.assertRaises(TriePrefixNotFoundError):
            self.model.get_next_tokens((99, 100))

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_generate_next_token_ideal(self) -> None:
        """
        Ideal generate_next_token scenario.
        """
        self.model.build()
        next_tokens = self.model.generate_next_token((1, 2))
        self.assertIsInstance(next_tokens, dict)

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_generate_next_token_invalid_input(self) -> None:
        """
        Invalid inputs for generate_next_token scenario.
        """
        self.model.build()
        bad_inputs = [1, [None], {}, None, (), 1.1, True, "hey"]
        for bad_input in bad_inputs:
            result = self.model.generate_next_token(bad_input)
            self.assertIsNone(result)

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_generate_next_token_short_sequence(self) -> None:
        """
        generate_next_token with short sequence scenario.
        """
        self.model.build()
        result = self.model.generate_next_token((1,))
        self.assertIsNone(result)

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_get_n_gram_size_ideal(self) -> None:
        """
        Ideal get_n_gram_size scenario.
        """
        self.assertEqual(self.model.get_n_gram_size(), 3)

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_get_node_by_prefix_ideal(self) -> None:
        """
        Ideal get_node_by_prefix scenario.
        """
        self.model.build()
        node = self.model.get_node_by_prefix((1, 2))
        self.assertIsNotNone(node)

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_update_ideal(self) -> None:
        """
        Ideal update scenario.
        """
        self.model.build()
        initial_root_children = len(self.model._root.get_children())

        new_corpus = ((3, 4, 5),)
        self.model.update(new_corpus)

        self.assertTrue(len(self.model._root.get_children()) >= initial_root_children)

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_collect_all_ngrams_ideal(self) -> None:
        """
        Ideal _collect_all_ngrams scenario.
        """
        self.model.build()
        all_ngrams = self.model._collect_all_ngrams()
        self.assertIsInstance(all_ngrams, tuple)
        self.assertTrue(len(all_ngrams) > 0)

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_collect_frequencies_ideal(self) -> None:
        """
        Ideal _collect_frequencies scenario.
        """
        self.model.build()
        node = self.model.get_node_by_prefix((1, 2))
        frequencies = self.model._collect_frequencies(node)
        self.assertIsInstance(frequencies, dict)

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_fill_frequencies_ideal(self) -> None:
        """
        Ideal _fill_frequencies scenario.
        """
        self.model.build()
        all_ngrams = self.model._collect_all_ngrams()
        self.model._fill_frequencies(all_ngrams)

        node = self.model.get_node_by_prefix((1, 2, 3))
        self.assertIsNotNone(node.get_frequency())

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_generate_next_token_prefix_not_found(self) -> None:
        """
        Scenario for generate_next_token when prefix is not found.
        """
        encoded_corpus = ((1, 2, 3), (1, 2, 4))
        model = NGramTrieLanguageModel(encoded_corpus, 3)
        model.build()

        next_tokens = model.generate_next_token((99, 100))
        self.assertEqual(next_tokens, {})

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_update_empty_initial_corpus(self) -> None:
        """
        Scenario for update when initial corpus is empty.
        """
        model = NGramTrieLanguageModel(None, 2)

        new_corpus = ((1, 2), (3, 4))
        model.update(new_corpus)

        self.assertEqual(model._encoded_corpus, new_corpus)

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_collect_all_ngrams_skips_none_data_children(self) -> None:
        """
        Scenario for _collect_all_ngrams for children with None data.
        """
        encoded_corpus = ((1, 2), (3, 4))
        model = NGramTrieLanguageModel(encoded_corpus, 2)
        model.build()

        with mock.patch.object(model._root, "get_children") as mock_get_children:
            child_data = mock.MagicMock()
            child_data.get_data.return_value = 1
            child_data.get_children.return_value = []

            child_none = mock.MagicMock()
            child_none.get_data.return_value = None
            child_none.get_children.return_value = []

            mock_get_children.return_value = [child_data, child_none]

            model._collect_all_ngrams()

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_collect_frequencies_skips_none_data_children(self) -> None:
        """
        Scenario for _collect_frequencies for children with None data.
        """
        encoded_corpus = ((1, 2), (3, 4))
        model = NGramTrieLanguageModel(encoded_corpus, 2)
        model.build()

        mock_node = mock.MagicMock()

        child_data = mock.MagicMock()
        child_data.get_data.return_value = 1
        child_data.get_frequency.return_value = 0.5

        child_none = mock.MagicMock()
        child_none.get_data.return_value = None

        mock_node.get_children.return_value = [child_data, child_none]

        frequencies = model._collect_frequencies(mock_node)

        self.assertEqual(frequencies, {1: 0.5})

    @pytest.mark.lab_4_auto_completion
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_get_next_tokens_leaf_node(self) -> None:
        """
        Leaf node for get_next_tokens scenario (no children).
        """
        encoded_corpus = ((1, 2, 3),)
        model = NGramTrieLanguageModel(encoded_corpus, 3)
        model.build()

        next_tokens = model.get_next_tokens((1, 2, 3))
        self.assertEqual(next_tokens, {})
