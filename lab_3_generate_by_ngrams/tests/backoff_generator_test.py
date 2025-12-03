# pylint: disable=protected-access
"""
Checks the third lab's BackOffGenerator class
"""
import unittest
from pathlib import Path
from unittest import mock

import pytest

from lab_3_generate_by_ngrams.main import BackOffGenerator, NGramLanguageModelReader, TextProcessor


class BackOffGeneratorTest(unittest.TestCase):
    """
    Tests BackOffGenerator class functionality
    """

    def setUp(self) -> None:
        """
        Setup of BackOffGeneratorTest.
        """
        path_to_test_directory = Path(__file__).parent.parent
        with open(
            path_to_test_directory / "assets" / "Anna Karenina - Chapter 1.txt",
            "r",
            encoding="utf-8",
        ) as text_file:
            text = text_file.read()
        self.backoff_processor = TextProcessor("_")
        self.backoff_processor.encode(text)

        reader = NGramLanguageModelReader(path_to_test_directory / "assets" / "en.json", "_")
        models = [reader.load(n_gram_size) for n_gram_size in (2, 3, 4)]
        self.language_models = tuple(model for model in models if model is not None)
        for language_model in self.language_models:
            language_model.build()

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark10
    def test_backoff_generator_fields(self) -> None:
        """
        Checks if BackOffGenerator fields are created correctly
        """
        backoff_text_generator = BackOffGenerator(self.language_models, self.backoff_processor)
        self.assertEqual(self.backoff_processor, backoff_text_generator._text_processor)
        expected = {
            model.get_n_gram_size(): model
            for model in sorted(
                self.language_models, key=lambda model: model.get_n_gram_size(), reverse=True
            )
        }
        self.assertEqual(expected, backoff_text_generator._language_models)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark10
    def test_backoff_generator_get_next_token(self) -> None:
        """
        Checks BackOffGenerator _get_next_token method ideal scenario
        """
        backoff_text_generator = BackOffGenerator(self.language_models, self.backoff_processor)
        expected = {0: 1.0}
        actual = backoff_text_generator._get_next_token((4, 9, 7))
        self.assertEqual(expected, actual)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark10
    def test_backoff_generator_run_empty_get_next_token(self) -> None:
        """
        Checks BackOffGenerator run method with empty _get_next_token return value
        """
        backoff_text_generator = BackOffGenerator(self.language_models, self.backoff_processor)
        backoff_text_generator._get_next_token = mock.Mock(return_value=[])
        actual = backoff_text_generator.run(50, "The")
        expected = "The."
        self.assertEqual(expected, actual)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark10
    def test_backoff_generator_get_next_token_none_generate_next_token(self) -> None:
        """
        Checks BackOffGenerator _get_next_token method with None
        as generate_next_token return value
        """
        backoff_text_generator = BackOffGenerator(self.language_models, self.backoff_processor)
        for index in (2, 3, 4):
            backoff_text_generator._language_models[index].generate_next_token = (
                lambda sequence: None
            )
        actual = backoff_text_generator._get_next_token((4, 9, 7))
        expected = None
        self.assertEqual(expected, actual)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark10
    def test_backoff_generator_get_next_token_empty_generate_next_token(self) -> None:
        """
        Checks BackOffGenerator _get_next_token method with empty generate_next_token
        """
        backoff_text_generator = BackOffGenerator(self.language_models, self.backoff_processor)
        for index in (2, 3, 4):
            backoff_text_generator._language_models[index].generate_next_token = lambda sequence: {}
        actual = backoff_text_generator._get_next_token((4, 9, 7))
        expected = None
        self.assertEqual(expected, actual)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark10
    def test_backoff_generator_get_next_token_without_models(self) -> None:
        """
        Checks BackOffGenerator _get_next_token method without models
        """
        backoff_text_generator = BackOffGenerator((), self.backoff_processor)
        actual = backoff_text_generator._get_next_token((4, 9, 7))
        expected = None
        self.assertEqual(expected, actual)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark10
    def test_backoff_generator_get_next_token_invalid_input(self) -> None:
        """
        Checks BackOffGenerator _get_next_token method with invalid inputs
        """
        backoff_text_generator = BackOffGenerator(self.language_models, self.backoff_processor)
        bad_inputs = [1, [None], {}, None, (), 1.1, True, "string"]
        expected = None
        for bad_input in bad_inputs:
            actual = backoff_text_generator._get_next_token(bad_input)
            self.assertEqual(expected, actual)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark10
    def test_backoff_generator_run(self) -> None:
        """
        Checks BackOffGenerator run method ideal scenario
        """
        backoff_text_generator = BackOffGenerator(self.language_models, self.backoff_processor)
        actual = backoff_text_generator.run(50, "The")
        self.assertEqual("The wsa wsa wsa wsa wsa wsa wsa wsa wsa wsa wsa wsa w.", actual)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark10
    def test_backoff_generator_run_invalid_input(self) -> None:
        """
        Checks BackOffGenerator run method with invalid inputs
        """
        backoff_text_generator = BackOffGenerator(self.language_models, self.backoff_processor)
        bad_len_inputs = [[None], {}, None, (), 1.1, "string"]
        bad_prompt_inputs = [1, [None], {}, None, (), 1.1, True, ""]
        expected = None
        for bad_input in bad_len_inputs:
            actual = backoff_text_generator.run(bad_input, "The")
            self.assertEqual(expected, actual)

        for bad_input in bad_prompt_inputs:
            actual = backoff_text_generator.run(50, bad_input)
            self.assertEqual(expected, actual)
