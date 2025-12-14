"""
Checks the third lab's BeamSearcher class
"""

# pylint: disable=duplicate-code, protected-access

import unittest
from pathlib import Path

import pytest

from lab_3_generate_by_ngrams.main import BeamSearcher, NGramLanguageModel, TextProcessor


class BeamSearcherTest(unittest.TestCase):
    """
    Tests BeamSearcher class functionality
    """

    def setUp(self) -> None:
        """
        Setup of BeamSearcherTest.
        """
        path_to_test_directory = Path(__file__).parent.parent
        with open(
            path_to_test_directory / "assets" / "Anna Karenina - Chapter 1.txt",
            "r",
            encoding="utf-8",
        ) as text_file:
            text = text_file.read()
        text_processor = TextProcessor("_")
        self.model = NGramLanguageModel(text_processor.encode(text), 3)
        self.model.build()

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_beam_searcher_fields(self) -> None:
        """
        Checks if BeamSearcher fields are created correctly
        """
        beam_search = BeamSearcher(3, self.model)
        self.assertEqual(3, beam_search._beam_width)
        self.assertEqual(self.model, beam_search._model)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_beam_searcher_get_next_token(self) -> None:
        """
        Checks BeamSearcher get_next_token method ideal scenario
        """
        beam_search = BeamSearcher(3, self.model)
        expected = [(0, 0.6617), (3, 0.1955), (6, 0.0376)]
        actual = beam_search.get_next_token((4, 9, 7))
        for expected_tuple, actual_tuple in zip(expected, actual):
            self.assertEqual(expected_tuple[0], actual_tuple[0])
            self.assertAlmostEqual(expected_tuple[1], actual_tuple[1], places=4)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_beam_searcher_get_next_token_invalid_input(self) -> None:
        """
        Checks BeamSearcher get_next_token method with invalid inputs
        """
        beam_search = BeamSearcher(3, self.model)
        bad_inputs = [1, [None], {}, None, (), 1.1, True, "string"]
        expected = None
        for bad_input in bad_inputs:
            actual = beam_search.get_next_token(bad_input)
            self.assertEqual(expected, actual)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_beam_searcher_get_next_token_none_generate_next_token(self) -> None:
        """
        Checks BeamSearcher run method with None
        as generate_next_token return value
        """
        beam_search = BeamSearcher(3, self.model)
        beam_search._model.generate_next_token = lambda sequence: None
        actual = beam_search.get_next_token((4, 9, 7))
        expected = None
        self.assertEqual(expected, actual)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_beam_searcher_get_next_token_empty_generate_next_token(self) -> None:
        """
        Checks BeamSearcher run method with empty generate_next_token
        """
        beam_search = BeamSearcher(3, self.model)
        beam_search._model.generate_next_token = lambda sequence: {}
        actual = beam_search.get_next_token((4, 9, 7))
        expected = []
        self.assertEqual(expected, actual)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_beam_searcher_continue_sequence(self) -> None:
        """
        Checks BeamSearcher continue_sequence method ideal scenario
        """
        beam_search = BeamSearcher(3, self.model)
        expected = {(4, 9, 7, 0): 4.5469, (4, 9, 7, 3): 5.9915, (4, 9, 7, 13): 7.4186}
        actual = beam_search.continue_sequence(
            (4, 9, 7), [(0, 0.0106), (3, 0.0025), (13, 0.0006)], {(4, 9, 7): 0.0}
        )
        for expected_tuple, actual_tuple in zip(expected, actual):
            self.assertEqual(expected_tuple[0], actual_tuple[0])
            self.assertAlmostEqual(expected_tuple[1], actual_tuple[1], places=4)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_beam_searcher_continue_sequence_invalid_input(self) -> None:
        """
        Checks BeamSearcher continue_sequence method with invalid inputs
        """
        beam_search = BeamSearcher(3, self.model)
        bad_sequence_inputs = [[None], 1, True, {}, None, (), 1.1, "string"]
        bad_next_tokens_inputs = [1, True, {}, None, [], 1.1, "string"]
        long_next_token_input = [(0, 0.0106), (3, 0.0025), (13, 0.0006), (20, 0.0001)]
        bad_sequence_candidates_input = [[None], 1, True, {}, None, (), 1.1, "string"]
        expected = None
        for bad_input in bad_sequence_inputs:
            actual = beam_search.continue_sequence(
                bad_input, [(0, 0.0106), (3, 0.0025), (13, 0.0006)], {(4, 9, 7): 0.0}
            )

            self.assertEqual(expected, actual)

        for bad_input in bad_next_tokens_inputs:
            actual = beam_search.continue_sequence((4, 9, 7), bad_input, {(4, 9, 7): 0.0})
            self.assertEqual(expected, actual)

        for bad_input in bad_sequence_candidates_input:
            actual = beam_search.continue_sequence(
                (4, 9, 7), [(0, 0.0106), (3, 0.0025), (13, 0.0006)], bad_input
            )

            self.assertEqual(expected, actual)

        actual_with_long_token = beam_search.continue_sequence(
            (4, 9, 7), long_next_token_input, {(4, 9, 7): 0.0}
        )
        self.assertEqual(expected, actual_with_long_token)
        actual_without_sequence_in_continue = beam_search.continue_sequence(
            (4, 9, 7), [(0, 0.0106), (3, 0.0025), (13, 0.0006)], {(2, 2, 2): 0.0}
        )
        self.assertEqual(expected, actual_without_sequence_in_continue)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_beam_searcher_prune_sequence_candidates(self) -> None:
        """
        Checks BeamSearcher prune_sequence_candidates method ideal scenario
        """
        beam_search = BeamSearcher(3, self.model)
        expected = {(4, 9, 7, 0): 4.551, (4, 9, 7, 3): 5.9934, (4, 9, 7, 13): 7.4597}
        actual = beam_search.prune_sequence_candidates(
            {(4, 9, 7, 0): 4.551, (4, 9, 7, 3): 5.9934, (4, 9, 7, 13): 7.4597}
        )
        self.assertEqual(expected, actual)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_beam_searcher_prune_sequence_candidates_invalid_inputs(self) -> None:
        """
        Checks BeamSearcher prune_sequence_candidates method with invalid inputs
        """
        beam_search = BeamSearcher(3, self.model)
        bad_inputs = [1, [None], {}, None, (), 1.1, True, "string"]
        expected = None
        for bad_input in bad_inputs:
            actual = beam_search.prune_sequence_candidates(bad_input)
            self.assertEqual(expected, actual)
