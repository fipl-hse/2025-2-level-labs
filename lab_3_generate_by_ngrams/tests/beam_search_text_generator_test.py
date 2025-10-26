# pylint: disable=protected-access
"""
Checks the third lab's BeamSearchTextGenerator class
"""

# pylint:disable=duplicate-code

import unittest
from pathlib import Path

import pytest

from lab_3_generate_by_ngrams.main import BeamSearchTextGenerator, NGramLanguageModel, TextProcessor


class BeamSearchTextGeneratorTest(unittest.TestCase):
    """
    Tests BeamSearchTextGenerator class functionality
    """

    def setUp(self) -> None:
        """
        Setup for BeamSearchTextGeneratorTest
        """
        path_to_test_directory = Path(__file__).parent.parent
        with open(
            path_to_test_directory / "assets" / "Anna Karenina - Chapter 1.txt",
            "r",
            encoding="utf-8",
        ) as text_file:
            text = text_file.read()
        self.text_processor = TextProcessor("_")
        self.language_model = NGramLanguageModel(self.text_processor.encode(text), 3)
        self.language_model.build()

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_beam_search_text_generator_fields(self) -> None:
        """
        Checks if BeamSearchTextGenerator fields are created correctly
        """
        beam_text_generator = BeamSearchTextGenerator(
            self.language_model, self.text_processor, beam_width=3
        )
        self.assertEqual(self.language_model, beam_text_generator._language_model)
        self.assertEqual(self.text_processor, beam_text_generator._text_processor)
        self.assertEqual(3, beam_text_generator._beam_width)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_beam_search_text_generator_run(self) -> None:
        """
        Checks BeamSearchTextGenerator run method ideal scenario
        """
        beam_text_generator = BeamSearchTextGenerator(
            self.language_model, self.text_processor, beam_width=3
        )
        actual = beam_text_generator.run("The", 70)
        self.assertEqual(
            "The his the his the his the his the his the his the his the his the his t.", actual
        )

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_beam_search_text_generator_run_invalid_input(self) -> None:
        """
        Checks BeamSearchTextGenerator run method with invalid inputs
        """
        beam_text_generator = BeamSearchTextGenerator(
            self.language_model, self.text_processor, beam_width=3
        )
        bad_seq_len_inputs = [[None], {}, None, (), 1.1, "string", 0]
        expected = None
        for bad_input in bad_seq_len_inputs:
            actual = beam_text_generator.run("The", bad_input)
            self.assertEqual(expected, actual)

        bad_prompt_inputs = [[None], 1, True, {}, None, (), 1.1, ""]
        expected = None
        for bad_input in bad_prompt_inputs:
            actual = beam_text_generator.run(bad_input, 70)
            self.assertEqual(expected, actual)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_beam_search_text_generator_run_none_encode(self) -> None:
        """
        Checks BeamSearchTextGenerator run method with None as encode return value
        """
        beam_text_generator = BeamSearchTextGenerator(
            self.language_model, self.text_processor, beam_width=3
        )
        beam_text_generator._text_processor.encode = lambda x: None
        actual = beam_text_generator.run("The", 70)
        expected = None
        self.assertEqual(expected, actual)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_beam_search_text_generator_run_none_get_next_token(self) -> None:
        """
        Checks BeamSearchTextGenerator run method with None
        as _get_next_token return value
        """
        beam_text_generator = BeamSearchTextGenerator(
            self.language_model, self.text_processor, beam_width=3
        )
        beam_text_generator._get_next_token = lambda sequence_to_continue: None
        actual = beam_text_generator.run("The", 70)
        expected = None
        self.assertEqual(expected, actual)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_beam_search_text_generator_run_none_continue_sequence(self) -> None:
        """
        Checks BeamSearchTextGenerator run method with None
        as _get_next_tokens return value
        """
        beam_text_generator = BeamSearchTextGenerator(
            self.language_model, self.text_processor, beam_width=3
        )
        beam_text_generator.beam_searcher.continue_sequence = (
            lambda sequence, next_tokens, sequence_candidates: None
        )
        actual = beam_text_generator.run("The", 70)
        expected = "The."
        self.assertEqual(expected, actual)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_beam_search_text_generator_run_none_prune_sequence_candidates(self) -> None:
        """
        Checks BeamSearchTextGenerator run method with None
        as prune_sequence_candidates return value
        """
        beam_text_generator = BeamSearchTextGenerator(
            self.language_model, self.text_processor, beam_width=3
        )
        beam_text_generator.beam_searcher.prune_sequence_candidates = (
            lambda sequence_candidates: None
        )
        actual = beam_text_generator.run("The", 70)
        expected = None
        self.assertEqual(expected, actual)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_beam_search_text_generator_get_next_tokens(self) -> None:
        """
        Checks BeamSearchTextGenerator _get_next_tokens method ideal scenario
        """
        beam_text_generator = BeamSearchTextGenerator(
            self.language_model, self.text_processor, beam_width=3
        )
        expected = [(0, 0.6617), (3, 0.1955), (6, 0.0376)]
        actual = beam_text_generator._get_next_token((4, 9, 7))
        for expected_tuple, actual_tuple in zip(expected, actual):
            self.assertEqual(expected_tuple[0], actual_tuple[0])
            self.assertAlmostEqual(expected_tuple[1], actual_tuple[1], places=4)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_beam_search_text_generator_get_next_tokens_invalid_input(self) -> None:
        """
        Checks BeamSearchTextGenerator _get_next_tokens method with invalid inputs
        """
        beam_text_generator = BeamSearchTextGenerator(
            self.language_model, self.text_processor, beam_width=3
        )
        expected = None
        bad_inputs = [1, [None], {}, None, (), 1.1, "string", 0]
        for bad_input in bad_inputs:
            actual = beam_text_generator._get_next_token(bad_input)
            self.assertEqual(expected, actual)
