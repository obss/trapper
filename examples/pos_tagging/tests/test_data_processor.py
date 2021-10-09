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
        add_prefix_space=True
    )


def test_data_processor(get_raw_dataset, args):
    expected_sentence = "The European Commission said on Thursday it disagreed with German advice to consumers to shun British lamb until scientists determine whether mad cow disease can be transmitted to sheep."
    if args.is_tokenizer_uncased:
        expected_sentence = expected_sentence.lower()
    data_processor = ExampleConll2003PosTaggingDataProcessor(args.tokenizer)
    raw_dataset = get_raw_dataset(
        path="conll2003_test_fixture", split="train"
    )
    processed_instance = raw_dataset.map(data_processor)[0]
    assert (args.tokenizer.decode(processed_instance["tokens"]).lstrip() ==
            expected_sentence)
    assert len(processed_instance["tokens"]) == len(processed_instance["pos_tags"])
