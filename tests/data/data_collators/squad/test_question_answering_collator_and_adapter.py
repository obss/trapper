import pytest
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from trapper.data.data_adapters.question_answering_adapter import (
    DataAdapterForQuestionAnswering,
)
from trapper.data.data_collator import InputBatch
from trapper.data.data_processors.squad import SquadQuestionAnsweringDataProcessor
from trapper.data.tokenizers import QuestionAnsweringTokenizerWrapper


@pytest.fixture(scope="module")
def data_collator_args(create_data_collator_args):
    return create_data_collator_args(
        tokenizer_factory=QuestionAnsweringTokenizerWrapper,
        train_batch_size=2,
        validation_batch_size=1,
        tokenizer_model_name="roberta-base",
        task_type="question_answering",
        is_distributed=False,
    )


@pytest.fixture(scope="module")
def processed_dataset(
    get_raw_dataset, data_collator_args, get_hf_datasets_fixture_path
):
    data_processor = SquadQuestionAnsweringDataProcessor(
        data_collator_args.tokenizer_wrapper
    )
    raw_dataset = get_raw_dataset(
        path=get_hf_datasets_fixture_path("squad_qa_test_fixture")
    )
    return raw_dataset.map(data_processor)


@pytest.fixture(scope="module")
def adapted_dataset(processed_dataset, data_collator_args):
    data_adapter = DataAdapterForQuestionAnswering(
        data_collator_args.tokenizer_wrapper
    )
    return processed_dataset.map(data_adapter)


@pytest.fixture(scope="module")
def qa_data_collator(make_data_collator, data_collator_args):
    return make_data_collator(data_collator_args)


@pytest.mark.parametrize(
    ["split", "expected_batch_size", "expected_dataset_size"],
    [
        ("train", 2, 3),
        ("validation", 1, 6),
    ],
)
def test_data_sizes(
    qa_data_collator,
    make_sequential_sampler,
    adapted_dataset,
    split,
    data_collator_args,
    expected_batch_size,
    expected_dataset_size,
):
    dataset_split = adapted_dataset[split]
    sampler = make_sequential_sampler(
        is_distributed=data_collator_args.is_distributed, dataset=dataset_split
    )
    loader = DataLoader(
        dataset_split,
        batch_size=getattr(data_collator_args, f"{split}_batch_size"),
        sampler=sampler,
        collate_fn=qa_data_collator,
    )
    assert loader.batch_size == expected_batch_size
    assert len(loader) == expected_dataset_size


@pytest.fixture(scope="module")
def collated_batch(qa_data_collator, adapted_dataset):
    return qa_data_collator.build_model_inputs(adapted_dataset["validation"])


@pytest.mark.parametrize(
    ["index", "expected_question"],
    [
        (0, "Which NFL team represented the AFC at Super Bowl 50?"),
        (1, "Which NFL team represented the NFC at Super Bowl 50?"),
        (2, "Where did Super Bowl 50 take place?"),
    ],
)
def test_batch_content(
    data_collator_args, processed_dataset, collated_batch, index, expected_question
):
    if data_collator_args.is_tokenizer_uncased:
        expected_question = expected_question.lower()
    validate_target_question_positions_using_decoded_tokens(
        expected_question,
        index,
        data_collator_args.tokenizer_wrapper.tokenizer,
        collated_batch,
    )

    instance = processed_dataset["validation"][index]
    token_type_ids = collated_batch["token_type_ids"][index]
    validate_token_type_ids(token_type_ids, instance)
    validate_attention_mask(collated_batch)


def validate_target_question_positions_using_decoded_tokens(
    expected_question,
    index,
    tokenizer: PreTrainedTokenizerBase,
    input_batch: InputBatch,
):
    input_ids = input_batch["input_ids"][index]
    question_start = -sum(input_batch["token_type_ids"][index])
    question_end = -1  # EOS
    assert (
        tokenizer.decode(input_ids[question_start:question_end]).lstrip()
        == expected_question
    )


def validate_token_type_ids(token_type_ids, instance):
    question_len = len(instance["question"])
    context_end = len(instance["context"]) + 2  # BOS, EOS
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
