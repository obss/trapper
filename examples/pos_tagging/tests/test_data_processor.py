from typing import Dict, List

import pytest

from examples.pos_tagging.src.data_processor import (
    ExampleConll2003PosTaggingDataProcessor,
)
from examples.pos_tagging.src.tokenizer import ExamplePosTaggingTokenizer


@pytest.fixture(scope="module")
def args(create_data_processor_args):
    return create_data_processor_args(
        tokenizer_cls=ExamplePosTaggingTokenizer,
        tokenizer_model_name="roberta-base",
        model_max_sequence_length=512,
    )


@pytest.fixture(scope="module")
def data_processor(args):
    return ExampleConll2003PosTaggingDataProcessor(args.tokenizer)


@pytest.fixture(scope="module")
def conlll03_dev_dataset(get_raw_dataset, data_processor):
    raw_dataset = get_raw_dataset(
        path="conll2003_test_fixture", split="validation"
    )
    return raw_dataset.map(data_processor)


@pytest.mark.parametrize(
    ["index", "expected_tags"],
    [
        (0, ["Denver Broncos", "Carolina Panthers", "Santa Clara, California"]),
        (1, ["Saint Bernadette Soubirous", "a golden statue of the Virgin Mary"]),
        (2, ["1882"]),
    ],
)
def test_data_processor(conlll03_dev_dataset, args, index, expected_tags):


    if args.is_tokenizer_uncased:
        expected_answers = [answer.lower() for answer in expected_tags]
    processed_instance = conlll03_dev_dataset[index]
    predicted_answer_texts = _reconstruct_tags(
        processed_instance, args.tokenizer
    )
    assert predicted_answer_texts == expected_tags
