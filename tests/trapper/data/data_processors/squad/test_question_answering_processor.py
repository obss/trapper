from dataclasses import dataclass
from typing import Dict

import pytest

from trapper.data import DatasetLoader, SquadQuestionAnsweringDataProcessor
from trapper.data.tokenizers import QuestionAnsweringTokenizer


@dataclass
class Arguments:
    distributed: bool = False
    shuffle: bool = False
    model_type: str = "roberta-base"
    train_batch_size: int = 2
    dev_batch_size: int = 1


@pytest.fixture(scope="module")
def args():
    args = Arguments()
    if "uncased" in args.model_type:
        args.uncased = True
    else:
        args.uncased = False
    return args


@pytest.fixture(scope="module")
def tokenizer(args):
    return QuestionAnsweringTokenizer.from_pretrained(args.model_type)


@pytest.fixture
def data_processor(tokenizer):
    return SquadQuestionAnsweringDataProcessor(tokenizer)


@pytest.fixture
def dataset_loader(data_processor):
    return DatasetLoader(
        data_processor=data_processor, path="squad_qa_test_fixture"
    )


@pytest.fixture
def fixtures_dir(fixtures_root):
    return fixtures_root / "data/question_answering"


@pytest.fixture
def dev_dataset(dataset_loader, fixtures_dir):
    return dataset_loader.load("validation")


@pytest.mark.parametrize(
    ["index", "question"],
    [
        (0, "Which NFL team represented the AFC at Super Bowl 50?"),
        (1, "Which NFL team represented the NFC at Super Bowl 50?"),
        (2, "Where did Super Bowl 50 take place?"),
    ],
)
def test_data_processor(tokenizer, dev_dataset, args, index, question):
    if args.uncased:
        question = question.lower()
    assert get_question(dev_dataset[index], tokenizer) == question


def get_question(instance: Dict, tokenizer: QuestionAnsweringTokenizer) -> str:
    field_value = tokenizer.decode(instance["question"])
    return field_value.lstrip()
