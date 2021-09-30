from dataclasses import dataclass
from typing import Dict

import pytest

from trapper.data import DatasetReader, SquadQuestionAnsweringDataProcessor
from trapper.data.tokenizers import QuestionAnsweringTokenizer


@dataclass
class Arguments:
    model_type: str = "roberta-base"


@pytest.fixture(scope="module")
def args():
    args = Arguments()
    if "uncased" in args.model_type:
        args.uncased = True
    else:
        args.uncased = False
    return args


@pytest.fixture
def dev_dataset():
    dataset_reader = DatasetReader(path="squad_qa_test_fixture")
    return dataset_reader.read("validation")


@pytest.fixture
def data_processor(args):
    tokenizer = QuestionAnsweringTokenizer.from_pretrained(args.model_type)
    return SquadQuestionAnsweringDataProcessor(tokenizer)


@pytest.mark.parametrize(
    ["index", "question"],
    [
        (0, "Which NFL team represented the AFC at Super Bowl 50?"),
        (1, "Which NFL team represented the NFC at Super Bowl 50?"),
        (2, "Where did Super Bowl 50 take place?"),
    ],
)
def test_data_processor(data_processor, dev_dataset, args, index, question):
    if args.uncased:
        question = question.lower()
    processed_instance = data_processor(dev_dataset[index])
    assert decode_question(processed_instance, data_processor.tokenizer) == question


def decode_question(instance: Dict, tokenizer: QuestionAnsweringTokenizer) -> str:
    field_value = tokenizer.decode(instance["question"])
    return field_value.lstrip()
