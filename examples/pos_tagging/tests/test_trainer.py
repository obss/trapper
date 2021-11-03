import datasets
import pytest
from transformers import DistilBertForTokenClassification, DistilBertTokenizerFast

# needed for registering the data-related classes
# noinspection PyUnresolvedReferences
# pylint: disable=unused-import
import examples.pos_tagging.src
from trapper.common import Params
from trapper.data.data_collator import DataCollator
from trapper.training import TransformerTrainer, TransformerTrainingArguments
from trapper.training.optimizers import HuggingfaceAdamWOptimizer
from trapper.training.train import run_experiment_using_trainer


@pytest.fixture(scope="module")
def trainer_params(temp_output_dir, temp_result_dir, get_hf_datasets_fixture_path):
    params_dict = {
        "pretrained_model_name_or_path": "distilbert-base-uncased",
        "train_split_name": "train",
        "dev_split_name": "validation",
        "tokenizer_wrapper": {
            "type": "pos_tagging_example",
            "add_prefix_space": True
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
        "metric_handler": {"type": "pos-tagging"},
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
            "seed": 100
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


@pytest.fixture(scope="module")
def trainer(trainer_params) -> TransformerTrainer:
    return TransformerTrainer.from_params(trainer_params)


def test_trainer_fields(trainer):
    assert isinstance(trainer, TransformerTrainer)
    assert isinstance(trainer.model, DistilBertForTokenClassification)
    assert isinstance(trainer.args, TransformerTrainingArguments)
    assert isinstance(trainer.data_collator, DataCollator)
    assert isinstance(trainer.train_dataset, datasets.Dataset)
    assert isinstance(trainer.eval_dataset, datasets.Dataset)
    assert isinstance(trainer.tokenizer, DistilBertTokenizerFast)
    assert isinstance(trainer.optimizer, HuggingfaceAdamWOptimizer)


def test_trainer_can_train(trainer):
    run_experiment_using_trainer(trainer)
