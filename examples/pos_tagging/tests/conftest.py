import pytest
from src import POS_TAGGING_FIXTURES_ROOT

from trapper.common import Params

# noinspection PyUnresolvedReferences
# pylint: disable=unused-import
from trapper.common.testing_utils.pytest_fixtures import (
    create_data_collator_args,
    create_data_processor_args,
    get_raw_dataset,
    make_data_collator,
    make_sequential_sampler,
    temp_output_dir,
    temp_result_dir,
)

_HF_DATASETS_FIXTURES_ROOT = POS_TAGGING_FIXTURES_ROOT / "hf_datasets"


@pytest.fixture(scope="package")
def get_hf_datasets_fixture_path():
    def _get_hf_datasets_fixture_path(dataset: str) -> str:
        return str(_HF_DATASETS_FIXTURES_ROOT / dataset)

    return _get_hf_datasets_fixture_path


@pytest.fixture(scope="module")
def experiment_params(
    temp_output_dir, temp_result_dir, get_hf_datasets_fixture_path
):
    params_dict = {
        "pretrained_model_name_or_path": "distilbert-base-uncased",
        "train_split_name": "train",
        "dev_split_name": "validation",
        "tokenizer_wrapper": {
            "type": "pos_tagging_example",
            "add_prefix_space": True,
        },
        "dataset_loader": {
            "dataset_reader": {
                "path": get_hf_datasets_fixture_path("conll2003_test_fixture"),
            },
            "data_processor": {
                "type": "conll2003_pos_tagging_example",
                "model_max_sequence_length": 512,
            },
            "data_adapter": {"type": "conll2003_pos_tagging_example"},
        },
        "data_collator": {},
        "model_wrapper": {"type": "token_classification", "num_labels": 47},
        "compute_metrics": {"metric_params": "seqeval"},
        "metric_input_handler": {"type": "token-classification"},
        "label_mapper": {"type": "conll2003_pos_tagging_example"},
        "args": {
            "type": "default",
            "output_dir": temp_output_dir + "/checkpoints",
            "result_dir": temp_result_dir,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "logging_dir": temp_output_dir + "/logs",
            "no_cuda": True,
            "logging_steps": 1,
            "evaluation_strategy": "steps",
            "save_steps": 2,
            "label_names": ["labels"],
            "lr_scheduler_type": "linear",
            "warmup_steps": 2,
            "do_train": True,
            "do_eval": True,
            "save_total_limit": 1,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "seed": 100,
        },
        "optimizer": {
            "type": "huggingface_adamw",
            "weight_decay": 0.01,
            "parameter_groups": [
                [
                    ["bias", "LayerNorm\\\\.weight", "layer_norm\\\\.weight"],
                    {"weight_decay": 0},
                ]
            ],
            "lr": 5e-5,
            "eps": 1e-8,
        },
    }
    return Params(params_dict)
