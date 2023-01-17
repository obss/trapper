import datasets
import pytest

# needed for registering the data-related classes
# noinspection PyUnresolvedReferences
# pylint: disable=unused-import
import src
from transformers import DistilBertForTokenClassification, DistilBertTokenizerFast

from trapper.data.data_collator import DataCollator
from trapper.training import TransformerTrainer, TransformerTrainingArguments
from trapper.training.optimizers import HuggingfaceAdamWOptimizer
from trapper.training.train import run_experiment_using_trainer


@pytest.fixture(scope="module")
def trainer(experiment_params) -> TransformerTrainer:
    return TransformerTrainer.from_params(experiment_params)


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
