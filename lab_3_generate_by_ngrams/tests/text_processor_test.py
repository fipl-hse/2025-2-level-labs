# pylint: disable=protected-access,too-many-public-methods
"""
Checks the third lab's TextProcessor class
"""
import json
import unittest
from pathlib import Path
from unittest import mock

import pytest

from lab_3_generate_by_ngrams.main import TextProcessor


class TextProcessorTest(unittest.TestCase):
    """
    Tests TextProcessor class functionality
    """

    def setUp(self) -> None:
        """
        Setup for TextProcessorTest
        """
        path_to_test_directory = Path(__file__).parent.parent
        with open(
            path_to_test_directory / "assets" / "en_own.json", "r", encoding="utf-8"
        ) as json_file:
            self.content = json.load(json_file)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_text_processor_fields(self) -> None:
        """
        Checks if TextProcessor fields are created correctly
        """
        text_processor = TextProcessor("_")
        self.assertEqual("_", text_processor._end_of_word_token)
        self.assertEqual({"_": 0}, text_processor._storage)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_text_processor_tokenize(self) -> None:
        """
        Checks TextProcessor _tokenize method ideal scenario
        """
        text_processor = TextProcessor("_")
        expected = (
            "g",
            "p",
            "t",
            "_",
            "о",
            "п",
            "е",
            "р",
            "и",
            "р",
            "у",
            "е",
            "т",
            "_",
            "п",
            "р",
            "и",
            "б",
            "л",
            "и",
            "з",
            "и",
            "т",
            "е",
            "л",
            "ь",
            "н",
            "о",
            "_",
            "т",
            "р",
            "л",
            "н",
            "_",
            "п",
            "а",
            "р",
            "а",
            "м",
            "е",
            "т",
            "р",
            "о",
            "в",
            "_",
            "н",
            "а",
            "_",
            "у",
            "р",
            "о",
            "в",
            "н",
            "я",
            "х",
            "_",
            "ч",
            "т",
            "о",
            "_",
            "в",
            "_",
            "р",
            "а",
            "з",
            "_",
            "б",
            "о",
            "л",
            "ь",
            "ш",
            "е",
            "_",
            "ч",
            "е",
            "м",
            "_",
            "у",
            "_",
            "g",
            "p",
            "t",
            "_",
        )
        actual = text_processor._tokenize(
            "GPT-4 оперирует приблизительно 1,8 трлн "
            "параметров на 120 уровнях, что в 10 раз "
            "больше, чем у GPT-3."
        )
        self.assertEqual(expected, actual)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_text_processor_tokenize_numbers(self) -> None:
        """
        Checks TextProcessor _tokenize method only with numbers
        """
        text_processor = TextProcessor("_")
        expected = None
        actual = text_processor._tokenize("123456")
        self.assertEqual(expected, actual)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_text_processor_tokenize_invalid_input(self) -> None:
        """
        Checks TextProcessor _tokenize method with invalid inputs
        """
        text_processor = TextProcessor("_")
        bad_inputs = [(), [None], {}, None, 1, 1.1, True]
        expected = None
        for bad_input in bad_inputs:
            actual = text_processor._tokenize(bad_input)
            self.assertEqual(expected, actual)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_text_processor_put(self) -> None:
        """
        Checks TextProcessor _put method ideal scenario
        """
        text_processor = TextProcessor("_")
        text_processor._put("f")

        actual = text_processor._storage
        self.assertEqual({"_": 0, "f": 1}, actual)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_text_processor_put_invalid_input(self) -> None:
        """
        Checks TextProcessor _put method with invalid inputs
        """
        bad_inputs = [(), [None], {}, None, 1, 1.1, True, "абв"]
        for bad_input in bad_inputs:
            text_processor = TextProcessor("_")
            text_processor._put(bad_input)
            actual = text_processor._storage
            expected = {"_": 0}
            self.assertEqual(expected, actual)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_text_processor_get_id(self) -> None:
        """
        Checks TextProcessor get_id method ideal scenario
        """
        text_processor = TextProcessor("_")
        text_processor._put("a")

        actual = text_processor.get_id("a")
        self.assertEqual(1, actual)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_text_processor_get_id_invalid_input(self) -> None:
        """
        Checks TextProcessor get_id method with invalid inputs
        """
        text_processor = TextProcessor("_")
        expected = None
        bad_inputs = [(), [None], {}, None, 1, 1.1, True]
        for bad_input in bad_inputs:
            actual = text_processor.get_id(bad_input)
            self.assertEqual(expected, actual)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_text_processor_get_id_missing_value(self) -> None:
        """
        Checks TextProcessor get_id method with missing value
        """
        text_processor = TextProcessor("_")
        expected = None
        actual = text_processor.get_id("a")
        self.assertEqual(expected, actual)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_text_processor_get_token(self) -> None:
        """
        Checks TextProcessor get_token method ideal scenario
        """
        text_processor = TextProcessor("_")
        text_processor._put("а")

        actual = text_processor.get_token(1)
        self.assertEqual("а", actual)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_text_processor_get_token_invalid_input(self) -> None:
        """
        Checks TextProcessor get_token method with invalid inputs
        """
        text_processor = TextProcessor("_")
        bad_inputs = [(), [None], {}, None, "string", 1.1, True]
        expected = None
        for bad_input in bad_inputs:
            actual = text_processor.get_token(bad_input)
            self.assertEqual(expected, actual)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_text_processor_get_token_missing_value(self) -> None:
        """
        Checks TextProcessor get_token method with missing value
        """
        text_processor = TextProcessor("_")
        expected = None
        actual = text_processor.get_token(1)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_text_processor_encode(self) -> None:
        """
        Checks TextProcessor encode method ideal scenario
        """
        text_processor = TextProcessor("_")
        expected = (
            1,
            2,
            3,
            0,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            0,
            11,
            12,
            6,
            7,
            13,
            14,
            11,
            5,
            15,
            9,
            6,
            7,
            16,
            9,
            10,
            0,
            10,
            17,
            18,
            19,
            5,
            20,
            9,
            10,
            0,
            11,
            5,
            15,
            21,
            6,
            7,
            0,
            22,
            5,
            17,
            15,
            9,
            16,
            16,
            9,
            10,
            0,
            23,
            2,
            24,
            25,
            26,
            27,
            0,
            28,
            21,
            13,
            20,
            29,
            30,
            13,
            9,
            10,
            0,
            20,
            0,
            22,
            21,
            30,
            14,
            14,
            0,
            1,
            2,
            3,
            0,
        )
        actual = text_processor.encode(
            "GPT-4 — большая мультимодальная языковая модель,"
            " созданная OpenAI, четвёртая в серии GPT."
        )

        self.assertEqual(expected, actual)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_text_processor_encode_none_tokenize(self) -> None:
        """
        Checks TextProcessor encode method with None as _tokenize return value
        """
        text_processor = TextProcessor("_")
        expected = None
        with mock.patch.object(text_processor, "_tokenize", return_value=None):
            actual = text_processor.encode(
                "GPT-4 — большая мультимодальная языковая модель,"
                " созданная OpenAI, четвёртая в серии GPT."
            )
        self.assertEqual(expected, actual)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_text_processor_encode_none_get_id(self) -> None:
        """
        Checks TextProcessor encode method with None as get_id return value
        """
        text_processor = TextProcessor("_")
        expected = None
        with mock.patch.object(text_processor, "get_id", return_value=None):
            actual = text_processor.encode(
                "GPT-4 — большая мультимодальная языковая модель,"
                " созданная OpenAI, четвёртая в серии GPT."
            )
        self.assertEqual(expected, actual)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_text_processor_encode_invalid_input(self) -> None:
        """
        Checks TextProcessor encode method with invalid inputs
        """
        text_processor = TextProcessor("_")
        bad_inputs = [1, [None], {}, None, (), 1.1, True, ""]
        expected = None
        for bad_input in bad_inputs:
            actual = text_processor.encode(bad_input)
            self.assertEqual(expected, actual)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark10
    def test_text_processor_fill_from_ngrams(self) -> None:
        """
        Checks TextProcessor fill_from_ngrams method ideal scenario
        """
        text_processor = TextProcessor("_")
        text_processor.fill_from_ngrams(self.content)
        expected = {
            "_": 0,
            "a": 2,
            "b": 22,
            "c": 19,
            "d": 17,
            "e": 4,
            "f": 18,
            "g": 15,
            "h": 1,
            "i": 8,
            "j": 23,
            "k": 24,
            "l": 12,
            "m": 20,
            "n": 10,
            "o": 14,
            "p": 3,
            "q": 25,
            "r": 13,
            "s": 9,
            "t": 7,
            "u": 11,
            "v": 21,
            "w": 6,
            "x": 5,
            "y": 16,
            "z": 26,
        }
        self.assertEqual(expected, text_processor._storage)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark10
    def test_text_processor_fill_from_ngrams_invalid_inputs(self) -> None:
        """
        Checks TextProcessor fill_from_ngrams method with invalid inputs
        """
        text_processor = TextProcessor("_")
        bad_inputs = [1, [None], {}, None, (), 1.1, True, "string"]
        expected = {"_": 0}
        for bad_input in bad_inputs:
            text_processor.fill_from_ngrams(bad_input)
            self.assertEqual(expected, text_processor._storage)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_text_processor_decode_protected(self) -> None:
        """
        Checks TextProcessor _decode method ideal scenario
        """
        text_processor = TextProcessor("_")
        encoded = text_processor.encode(
            "GPT-4 — большая мультимодальная языковая модель,"
            " созданная OpenAI, четвёртая в серии GPT."
        )
        expected = (
            "g",
            "p",
            "t",
            "_",
            "б",
            "о",
            "л",
            "ь",
            "ш",
            "а",
            "я",
            "_",
            "м",
            "у",
            "л",
            "ь",
            "т",
            "и",
            "м",
            "о",
            "д",
            "а",
            "л",
            "ь",
            "н",
            "а",
            "я",
            "_",
            "я",
            "з",
            "ы",
            "к",
            "о",
            "в",
            "а",
            "я",
            "_",
            "м",
            "о",
            "д",
            "е",
            "л",
            "ь",
            "_",
            "с",
            "о",
            "з",
            "д",
            "а",
            "н",
            "н",
            "а",
            "я",
            "_",
            "o",
            "p",
            "e",
            "n",
            "a",
            "i",
            "_",
            "ч",
            "е",
            "т",
            "в",
            "ё",
            "р",
            "т",
            "а",
            "я",
            "_",
            "в",
            "_",
            "с",
            "е",
            "р",
            "и",
            "и",
            "_",
            "g",
            "p",
            "t",
            "_",
        )
        actual = text_processor._decode(encoded)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_text_processor_decode_protected_none_get_token(self) -> None:
        """
        Checks TextProcessor _decode method with None as get_token return value
        """
        text_processor = TextProcessor("_")
        encoded = text_processor.encode(
            "GPT-4 — большая мультимодальная языковая модель,"
            " созданная OpenAI, четвёртая в серии GPT."
        )
        expected = None
        with mock.patch.object(text_processor, "get_token", return_value=None):
            actual = text_processor._decode(encoded)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_text_processor_decode_protected_invalid_input(self) -> None:
        """
        Checks TextProcessor _decode method with invalid inputs
        """
        text_processor = TextProcessor("_")
        bad_inputs = [1, [None], {}, None, (), 1.1, True, "string"]
        expected = None
        for bad_input in bad_inputs:
            actual = text_processor._decode(bad_input)
            self.assertEqual(expected, actual)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_text_processor_postprocess_decoded_text(self) -> None:
        """
        Checks TextProcessor _postprocess_decoded_text method ideal scenario
        """
        text_processor = TextProcessor("_")
        encoded = text_processor.encode(
            "GPT-4 — большая мультимодальная языковая модель,"
            " созданная OpenAI, четвёртая в серии GPT."
        )
        protected_decoded = text_processor._decode(encoded)
        expected = (
            "Gpt большая мультимодальная языковая модель созданная openai четвёртая в серии gpt."
        )
        actual = text_processor._postprocess_decoded_text(protected_decoded)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_text_processor_postprocess_decoded_text_invalid_input(self) -> None:
        """
        Checks TextProcessor _postprocess_decoded_text method with invalid inputs
        """
        text_processor = TextProcessor("_")
        bad_inputs = [1, [None], {}, None, (), 1.1, True, "string"]
        expected = None
        for bad_input in bad_inputs:
            actual = text_processor._postprocess_decoded_text(bad_input)
            self.assertEqual(expected, actual)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_text_processor_decode(self) -> None:
        """
        Checks TextProcessor decode method ideal scenario
        """
        text_processor = TextProcessor("_")
        encoded = text_processor.encode(
            "GPT-4 — большая мультимодальная языковая модель,"
            " созданная OpenAI, четвёртая в серии GPT."
        )
        expected = (
            "Gpt большая мультимодальная языковая модель созданная openai четвёртая в серии gpt."
        )
        actual = text_processor.decode(encoded)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_text_processor_decode_none_decode_protected(self) -> None:
        """
        Checks TextProcessor decode method with None as _decode return value
        """
        text_processor = TextProcessor("_")
        encoded = text_processor.encode(
            "GPT-4 — большая мультимодальная языковая модель,"
            " созданная OpenAI, четвёртая в серии GPT."
        )
        expected = None
        with mock.patch.object(text_processor, "_decode", return_value=None):
            actual = text_processor.decode(encoded)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_text_processor_decode_bad_input(self) -> None:
        """
        Checks TextProcessor decode method with invalid inputs
        """
        text_processor = TextProcessor("_")
        bad_inputs = [1, [None], {}, None, (), 1.1, True, "string"]
        expected = None
        for bad_input in bad_inputs:
            actual = text_processor.decode(bad_input)
            self.assertEqual(expected, actual)

    @pytest.mark.lab_3_generate_by_ngrams
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_text_processor_get_end_of_word_token(self) -> None:
        """
        Checks TextProcessor get_end_of_word_token method ideal scenario
        """
        text_processor = TextProcessor("_")
        self.assertEqual("_", text_processor.get_end_of_word_token())
