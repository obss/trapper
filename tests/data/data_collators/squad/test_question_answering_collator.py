from dataclasses import dataclass
from typing import NamedTuple, Union

import pytest
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.sampler import SequentialSampler
from transformers.trainer_pt_utils import SequentialDistributedSampler

from trapper.data.data_collators import DataCollatorForQuestionAnswering, InputBatch
from trapper.data.dataset_readers import SquadQuestionAnsweringDatasetReader
from trapper.data.tokenizers import QuestionAnsweringTokenizer
from trapper.models.auto_wrappers import _TASK_TO_INPUT_FIELDS


class DataItems(NamedTuple):
    loader: DataLoader
    sampler: Union[DistributedSampler, SequentialDistributedSampler]


@dataclass
class Arguments:
    distributed: bool = False
    task_type: str = "question_answering"
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
        cache_file_prefix="test-squad-clue-extraction",
        cache_directory=tempdir,
    )


@pytest.fixture
def fixtures_dir(fixtures_root):
    return fixtures_root / "data/question_answering"


@pytest.fixture
def dev_dataset(dataset_reader, fixtures_dir):
    return dataset_reader.read(fixtures_dir / "squad_qa/dev.json")


@pytest.fixture
def train_dataset(dataset_reader, fixtures_dir):
    return dataset_reader.read(fixtures_dir / "squad_qa/train.json")


@pytest.fixture(scope="module")
def model_forward_params(args):
    return _TASK_TO_INPUT_FIELDS[args.task_type]


@pytest.fixture
def dataset_collator(tokenizer, model_forward_params):
    return DataCollatorForQuestionAnswering(tokenizer, model_forward_params)


def get_sequential_sampler(distributed_training, dataset):
    if distributed_training:
        return SequentialDistributedSampler(dataset)
    return SequentialSampler(dataset)


@pytest.fixture
def dev_data_items(dataset_collator, dev_dataset, args):
    sampler = get_sequential_sampler(args.distributed, dev_dataset)
    loader = DataLoader(
        dev_dataset,
        batch_size=args.dev_batch_size,
        sampler=sampler,
        collate_fn=dataset_collator,
    )
    return DataItems(loader=loader, sampler=sampler)


@pytest.fixture
def train_data_items(dataset_collator, train_dataset, args):
    sampler = get_sequential_sampler(args.distributed, train_dataset)
    loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        sampler=sampler,
        collate_fn=dataset_collator,
    )
    return DataItems(loader=loader, sampler=sampler)


def test_data_sizes(dev_data_items, train_data_items):
    train_loader = train_data_items.loader
    dev_loader = dev_data_items.loader
    assert train_loader.batch_size == 2
    assert len(train_loader) == 3
    assert dev_loader.batch_size == 1
    assert len(dev_loader) == 6


@pytest.fixture
def collated_batch(dataset_collator, dev_dataset):
    return dataset_collator.build_model_inputs(dev_dataset)


@pytest.mark.parametrize(
    ["index", "question"],
    [
        (0, "Which NFL team represented the AFC at Super Bowl 50?"),
        (1, "Which NFL team represented the NFC at Super Bowl 50?"),
        (2, "Where did Super Bowl 50 take place?"),
    ],
)
def test_batch_content(
    args, tokenizer, dev_dataset, index, question, collated_batch
):
    if args.uncased:
        question = question.lower()
    validate_target_question_positions_using_decoded_tokens(
        question, index, tokenizer, collated_batch
    )

    raw_instance = dev_dataset[index]
    token_type_ids = collated_batch["token_type_ids"][index]
    validate_token_type_ids(token_type_ids, raw_instance)
    validate_attention_mask(collated_batch)


def validate_target_question_positions_using_decoded_tokens(
    question, index, tokenizer, input_batch: InputBatch
):
    input_ids = input_batch["input_ids"][index]
    question_start = -sum(input_batch["token_type_ids"][index])
    question_end = -1  # EOS
    assert (
        tokenizer.decode(input_ids[question_start:question_end]).lstrip()
        == question
    )


def validate_token_type_ids(token_type_ids, raw_instance):
    question_len = len(raw_instance["question"])
    context_end = len(raw_instance["context"]) + 2  # BOS, EOS
    question_end = context_end + question_len + 1  # EOS

    # remaining context tokens
    assert all(token_type_id == 0 for token_type_id in token_type_ids[:context_end])

    # answer tokens at the end
    assert all(token_type_id == 1 for token_type_id in token_type_ids[context_end:])

    assert len(token_type_ids) == question_end


def validate_attention_mask(instance_batch: InputBatch):
    for input_ids, attention_mask in zip(
        instance_batch["input_ids"], instance_batch["attention_mask"]
    ):
        assert len(attention_mask) == len(input_ids)
        assert all(val == 1 for val in attention_mask)
