from dataclasses import dataclass
from typing import Dict

import pytest

from trapper.data.dataset_readers import SquadQuestionAnsweringDatasetReader
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
def dataset_reader(tokenizer, tempdir):
    return SquadQuestionAnsweringDatasetReader(
        tokenizer=tokenizer,
        apply_cache=True,
        cache_file_prefix="test-squad-question-answering",
        cache_directory=tempdir,
    )


@pytest.fixture
def fixtures_dir(fixtures_root):
    return fixtures_root / "data/question_answering"


@pytest.fixture
def dev_dataset(dataset_reader, fixtures_dir):
    return dataset_reader.read(fixtures_dir / "squad_qa/dev.json")


@pytest.mark.parametrize(
    ["index", "question"],
    [
        (0, "Which NFL team represented the AFC at Super Bow 50?"),
        (1, "Which NFL team represented the NFC at Super Bowl 50?"),
        (2, "Where did Super Bowl 50 take place?"),
    ],
)
def test_data_reader(tokenizer, dev_dataset, args, index, question):
    if args.uncased:
        question = question.lower()
    assert get_question(dev_dataset[index], tokenizer) == question


def get_question(instance: Dict, tokenizer: QuestionAnsweringTokenizer) -> str:
    field_value = tokenizer.decode(instance["question"])
    return field_value.lstrip()
