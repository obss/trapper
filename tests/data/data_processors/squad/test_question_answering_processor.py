from typing import Dict

import pytest

from trapper.data import SquadQuestionAnsweringDataProcessor
from trapper.data.tokenizers import QuestionAnsweringTokenizer


@pytest.fixture
def args(get_data_processor_args):
    return get_data_processor_args(
        tokenizer_cls=QuestionAnsweringTokenizer,
        model_type="roberta-base")


@pytest.fixture
def processed_dev_dataset(get_raw_dataset, args):
    data_processor = SquadQuestionAnsweringDataProcessor(args.tokenizer)
    raw_dataset = get_raw_dataset(path="squad_qa_test_fixture", split="validation")
    return raw_dataset.map(data_processor)


@pytest.mark.parametrize(
    ["index", "question"],
    [
        (0, "Which NFL team represented the AFC at Super Bowl 50?"),
        (1, "Which NFL team represented the NFC at Super Bowl 50?"),
        (2, "Where did Super Bowl 50 take place?"),
    ],
)
def test_data_processor(processed_dev_dataset, args, index, question):
    if args.uncased:
        question = question.lower()
    processed_instance = processed_dev_dataset[index]
    assert decode_question(processed_instance, args.tokenizer) == question


def decode_question(instance: Dict, tokenizer: QuestionAnsweringTokenizer
                    ) -> str:
    field_value = tokenizer.decode(instance["question"])
    return field_value.lstrip()
