from typing import Dict

import pytest
from transformers import PreTrainedTokenizerBase

from trapper.data import SquadQuestionAnsweringDataProcessor
from trapper.data.tokenizers import QuestionAnsweringTokenizerWrapper


@pytest.fixture(scope="module")
def args(create_data_processor_args):
    return create_data_processor_args(
        tokenizer_factory=QuestionAnsweringTokenizerWrapper,
        tokenizer_model_name="roberta-base")


@pytest.fixture(scope="module")
def processed_dev_dataset(get_raw_dataset, args):
    data_processor = SquadQuestionAnsweringDataProcessor(args.tokenizer_wrapper)
    raw_dataset = get_raw_dataset(path="squad_qa_test_fixture", split="validation")
    return raw_dataset.map(data_processor)


@pytest.mark.parametrize(
    ["index", "expected_question"],
    [
        (0, "Which NFL team represented the AFC at Super Bowl 50?"),
        (1, "Which NFL team represented the NFC at Super Bowl 50?"),
        (2, "Where did Super Bowl 50 take place?"),
    ],
)
def test_data_processor(processed_dev_dataset, args, index, expected_question):
    if args.is_tokenizer_uncased:
        expected_question = expected_question.lower()
    processed_instance = processed_dev_dataset[index]
    assert decode_question(processed_instance,
                           args.tokenizer_wrapper.tokenizer) == expected_question


def decode_question(instance: Dict, tokenizer: PreTrainedTokenizerBase
                    ) -> str:
    field_value = tokenizer.decode(instance["question"])
    return field_value.lstrip()
