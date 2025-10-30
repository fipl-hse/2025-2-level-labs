# pylint: disable=protected-access
"""
Checks the third lab's GreedyTextGenerator class
"""
import unittest

import pytest

from lab_3_generate_by_ngrams.main import GreedyTextGenerator, NGramLanguageModel, TextProcessor


class GreedyTextGeneratorTest(unittest.TestCase):
    """
    Tests GreedyTextGenerator class functionality
    """

    def setUp(self) -> None:
        """
        Setup for GreedyTextGeneratorTest
        """
        text = """Most unpleasant of all was the first minute when, on coming, happy and
        good-humored, from the theater, with a huge pear in his hand for his
        wife, he had not found his wife in the drawing-room, to his surprise
        had not found her in the study either, and saw her at last in her
        bedroom with the unlucky letter that revealed everything in her hand."""
        self.processor = TextProcessor("_")
        self.encoded = self.processor.encode(text)
        self.language_model = NGramLanguageModel(self.encoded, 3)
        self.language_model.build()

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_greedy_text_generator_fields(self) -> None:
        """
        Checks if GreedyTextGenerator fields are created correctly
        """
        greedy_text_generator = GreedyTextGenerator(self.language_model, self.processor)
        self.assertEqual(self.language_model, greedy_text_generator._model)
        self.assertEqual(self.processor, greedy_text_generator._text_processor)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_greedy_text_generator_run(self) -> None:
        """
        Checks GreedyTextGenerator run method ideal scenario
        """
        greedy_text_generator = GreedyTextGenerator(self.language_model, self.processor)
        actual = greedy_text_generator.run(4, "He")
        self.assertEqual("He had.", actual)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_greedy_text_generator_run_none_encode(self) -> None:
        """
        Checks GreedyTextGenerator run method with None as encode return value
        """
        greedy_text_generator = GreedyTextGenerator(self.language_model, self.processor)
        greedy_text_generator._text_processor.encode = lambda x: None
        actual = greedy_text_generator.run(4, "He")
        expected = None
        self.assertEqual(expected, actual)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_greedy_text_generator_run_none_generate_next_token(self) -> None:
        """
        Checks GreedyTextGenerator run method with None
        as generate_next_token return value
        """
        greedy_text_generator = GreedyTextGenerator(self.language_model, self.processor)
        greedy_text_generator._model.generate_next_token = lambda sequence: None
        actual = greedy_text_generator.run(4, "He")
        expected = "He."
        self.assertEqual(expected, actual)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_greedy_text_generator_run_empty_next_candidate(self) -> None:
        """
        Checks GreedyTextGenerator run method with empty next candidate
        """
        greedy_text_generator = GreedyTextGenerator(self.language_model, self.processor)
        greedy_text_generator._model.generate_next_token = lambda sequence: {}
        actual = greedy_text_generator.run(4, "He")
        expected = "He."
        self.assertEqual(expected, actual)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_greedy_text_generator_run_invalid_input(self) -> None:
        """
        Checks GreedyTextGenerator run method with invalid inputs
        """
        greedy_text_generator = GreedyTextGenerator(self.language_model, self.processor)
        expected = None
        seq_len_bad_input = [[None], {}, None, (), 1.1, "string"]
        prompt_bad_input = [1, [None], {}, None, (), 1.1, "", 0]
        for bad_input in seq_len_bad_input:
            actual = greedy_text_generator.run(bad_input, "He")
            self.assertEqual(expected, actual)

        for bad_input in prompt_bad_input:
            actual = greedy_text_generator.run(4, bad_input)
            self.assertEqual(expected, actual)
