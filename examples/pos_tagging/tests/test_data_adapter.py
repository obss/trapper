import pytest
from src.data.data_adapter import ExampleDataAdapterForPosTagging
from src.data.data_processor import ExampleConll2003PosTaggingDataProcessor
from src.data.tokenizer_wrapper import ExamplePosTaggingTokenizerWrapper

from trapper.common.constants import IGNORED_LABEL_ID
from trapper.data import InputBatch


@pytest.fixture(scope="module")
def data_collator_args(create_data_collator_args):
    return create_data_collator_args(
        task_type="token_classification",
        train_batch_size=1,
        validation_batch_size=1,
        is_distributed=False,
        model_max_sequence_length=512,
        tokenizer_factory=ExamplePosTaggingTokenizerWrapper,
        tokenizer_model_name="roberta-base",
        add_prefix_space=True,
    )


@pytest.fixture(scope="module")
def raw_conll03_postagging_dataset(get_raw_dataset, get_hf_datasets_fixture_path):
    return get_raw_dataset(
        path=get_hf_datasets_fixture_path("conll2003_test_fixture"), split="train"
    )


@pytest.fixture(scope="module")
def adapted_conll03_postagging_dataset(
    raw_conll03_postagging_dataset, data_collator_args
):
    data_adapter = ExampleDataAdapterForPosTagging(
        data_collator_args.tokenizer_wrapper
    )
    data_processor = ExampleConll2003PosTaggingDataProcessor(
        data_collator_args.tokenizer_wrapper
    )
    processed_dataset = raw_conll03_postagging_dataset.map(data_processor)
    return processed_dataset.map(data_adapter)


@pytest.fixture(scope="module")
def data_collator(make_data_collator, data_collator_args):
    return make_data_collator(data_collator_args)


def test_batch_content_on_squad_dev_dataset(
    raw_conll03_postagging_dataset,
    adapted_conll03_postagging_dataset,
    data_collator_args,
    data_collator,
):
    expected_sentence = "The European Commission said on Thursday it disagreed with German advice to consumers to shun British lamb until scientists determine whether mad cow disease can be transmitted to sheep."
    if data_collator_args.is_tokenizer_uncased:
        expected_sentence = expected_sentence.lower()

    collated_batch = data_collator.build_model_inputs(
        adapted_conll03_postagging_dataset
    )
    input_ids = collated_batch["input_ids"][0]
    labels = collated_batch["labels"][0]

    tokenizer = data_collator_args.tokenizer_wrapper.tokenizer
    decoded_sentence = tokenizer.decode(
        input_ids, skip_special_tokens=True
    ).lstrip()
    assert expected_sentence == decoded_sentence
    assert len(input_ids) == len(labels)

    encoding = tokenizer(expected_sentence, add_special_tokens=False)
    raw_pos_tags = [
        12,
        22,
        22,
        38,
        15,
        22,
        28,
        38,
        15,
        16,
        21,
        35,
        24,
        35,
        37,
        16,
        21,
        15,
        24,
        41,
        15,
        16,
        21,
        21,
        20,
        37,
        40,
        35,
        21,
        7,
    ]
    expected_labels = [raw_pos_tags[ind] for ind in encoding.word_ids()]
    expected_labels.insert(0, IGNORED_LABEL_ID)  # BOS
    expected_labels.append(IGNORED_LABEL_ID)  # EOS
    assert expected_labels == labels
    validate_attention_mask(collated_batch)


def validate_attention_mask(instance_batch: InputBatch):
    for input_ids, attention_mask in zip(
        instance_batch["input_ids"], instance_batch["attention_mask"]
    ):
        assert len(attention_mask) == len(input_ids)
        assert all(val == 1 for val in attention_mask)
