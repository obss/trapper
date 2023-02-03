from typing import Any, Dict

import datasets
import pytest
from transformers import (
    EvalPrediction,
    T5ForConditionalGeneration,
    T5Tokenizer,
    T5TokenizerFast,
)

from trapper.common import Params
from trapper.common.constants import IGNORED_LABEL_ID
from trapper.data import (
    DataAdapter,
    DataProcessor,
    IndexedInstance,
    TokenizerWrapper,
)
from trapper.data.data_collator import DataCollator
from trapper.metrics import MetricInputHandler
from trapper.training import TransformerTrainer, TransformerTrainingArguments
from trapper.training.optimizers import HuggingfaceAdamWOptimizer
from trapper.training.train import run_experiment_using_trainer
from trapper.training.trainer import Seq2SeqTransformerTrainer


@DataProcessor.register("dummy_conversational")
class DummyConversationalDataProcessor(DataProcessor):
    NUM_EXTRA_SPECIAL_TOKENS_IN_SEQUENCE = 2  # <bos> tokens <eos>

    def process(self, instance_dict: Dict[str, Any]) -> IndexedInstance:
        return self.text_to_instance(
            id_=instance_dict["id"],
            past_user_inputs="".join(instance_dict["past_user_inputs"]),
            generated_responses="".join(instance_dict["generated_responses"]),
        )

    def text_to_instance(
        self, id_: str, past_user_inputs: str, generated_responses: str = None
    ) -> IndexedInstance:
        instance = {"id": id_}
        indexed_past_user_inputs = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize(past_user_inputs)
        )
        if not indexed_past_user_inputs:
            instance["past_user_inputs"] = [-1]
            instance["generated_responses"] = [-1]
            instance["__discard_sample"] = True
            return instance

        instance["past_user_inputs"] = indexed_past_user_inputs
        if generated_responses is not None:
            instance["generated_responses"] = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(generated_responses)
            )
        return instance


@DataAdapter.register("dummy_conversational")
class DataAdapterForDummyConversational(DataAdapter):
    INPUT_TOKEN_TYPE_ID = 0
    OUTPUT_TOKEN_TYPE_ID = 1

    def __init__(
        self,
        tokenizer_wrapper: TokenizerWrapper,
        **kwargs,  # for the ignored args such as label mapper
    ):
        # self.answer_token_id = self._tokenizer.convert_tokens_to_ids(ANSWER_TOKEN)
        super().__init__(tokenizer_wrapper)
        self._eos_token_id = self._tokenizer.eos_token_id
        self._bos_token_id = self._tokenizer.bos_token_id

    def __call__(self, instance: IndexedInstance) -> IndexedInstance:
        input_ids = (
            [self._bos_token_id]
            + instance["past_user_inputs"]
            + [self._eos_token_id]
        )
        prompt_len = len(input_ids)
        input_ids.extend(instance["generated_responses"] + [self._eos_token_id])
        token_type_ids = [
            self.INPUT_TOKEN_TYPE_ID
            if i < prompt_len
            else self.OUTPUT_TOKEN_TYPE_ID
            for i in range(len(input_ids))
        ]
        labels = [IGNORED_LABEL_ID] * prompt_len
        labels.extend(instance["generated_responses"] + [self._eos_token_id])
        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "labels": labels,
        }


@MetricInputHandler.register("pass_through")
class PassThroughMetricInputHandler(MetricInputHandler):
    def __call__(self, eval_pred: EvalPrediction) -> EvalPrediction:
        if isinstance(eval_pred.predictions, tuple):
            eval_pred = EvalPrediction(
                # Models like T5 returns a tuple of (
                # logits, encoder_last_hidden_state) instead of only the logits
                predictions=eval_pred.predictions[0],
                label_ids=eval_pred.label_ids,
            )

        return super().__call__(eval_pred)


@pytest.fixture(scope="module")
def trainer_params(
    temp_output_dir, temp_result_dir, get_hf_datasets_fixture_path_from_root
):
    params_dict = {
        "type": "seq2seq",
        "pretrained_model_name_or_path": "t5-small",
        "train_split_name": "test",
        "dev_split_name": "test",
        "tokenizer_wrapper": {"type": "from_pretrained"},
        "dataset_loader": {
            "dataset_reader": {"path": "Narsil/conversational_dummy"},
            "data_processor": {"type": "dummy_conversational"},
            "data_adapter": {"type": "dummy_conversational"},
        },
        "data_collator": {},
        "model_wrapper": {"type": "seq2seq_lm"},
        "compute_metrics": {"metric_params": ["rouge"]},
        "metric_input_handler": {"type": "language-generation"},
        "metric_output_handler": {"type": "default"},
        "args": {
            "type": "seq2seq",
            "output_dir": temp_output_dir + "/checkpoints",
            "result_dir": temp_result_dir,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 3,
            "per_device_eval_batch_size": 2,
            "logging_dir": temp_output_dir + "/logs",
            "no_cuda": True,
            "logging_steps": 2,
            "evaluation_strategy": "steps",
            "save_steps": 3,
            "label_names": ["labels"],
            "lr_scheduler_type": "linear",
            "warmup_steps": 2,
            "do_train": True,
            "do_eval": True,
            "save_total_limit": 1,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
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
    assert isinstance(trainer, Seq2SeqTransformerTrainer)
    assert isinstance(trainer.model, T5ForConditionalGeneration)
    assert isinstance(trainer.args, TransformerTrainingArguments)
    assert isinstance(trainer.data_collator, DataCollator)
    assert isinstance(trainer.train_dataset, datasets.Dataset)
    assert isinstance(trainer.eval_dataset, datasets.Dataset)
    assert isinstance(trainer.tokenizer, T5Tokenizer) or isinstance(
        trainer.tokenizer, T5TokenizerFast
    )
    assert isinstance(trainer.optimizer, HuggingfaceAdamWOptimizer)


def test_trainer_can_train(trainer):
    run_experiment_using_trainer(trainer)
