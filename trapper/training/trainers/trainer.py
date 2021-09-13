import re
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.optim.lr_scheduler import LambdaLR
from transformers import PreTrainedModel
from transformers.trainer import Trainer as _Trainer
from transformers.trainer_utils import EvalPrediction

from trapper.common import Lazy, Registrable
from trapper.common.plugins import import_plugins
from trapper.common.utils import append_parent_docstr
from trapper.data import (
    DatasetReader,
    TransformerDataCollator,
    TransformerTokenizer,
)
from trapper.data.dataset_readers import IndexedDataset
from trapper.models import TransformerModel
from trapper.training.callbacks import TrainerCallback
from trapper.training.metrics import TransformerMetric
from trapper.training.optimizers import Optimizer
from trapper.training.training_args import TransformerTrainingArguments


@append_parent_docstr(parent_id=0)
class TransformerTrainer(_Trainer, Registrable):
    """
    `Trapper`'s default trainer that wraps the `Trainer` class from the
    `Transformers` library.

    Args:
        model ():
        args ():
        data_collator ():
        train_dataset ():
        eval_dataset ():
        tokenizer ():
        model_init ():
        compute_metrics ():
        callbacks ():
        optimizers ():
    """

    default_implementation = "default"

    def __init__(
        self,
        model: PreTrainedModel = None,
        args: TransformerTrainingArguments = None,
        data_collator: Optional[TransformerDataCollator] = None,
        train_dataset: Optional[IndexedDataset] = None,
        eval_dataset: Optional[IndexedDataset] = None,
        tokenizer: Optional[TransformerTokenizer] = None,
        model_init: Callable[[], TransformerModel] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, Optional[LambdaLR]] = (None, None),
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
        )

    @classmethod
    def from_partial_objects(
        cls,
        pretrained_model_name_or_path: str,
        train_file_path: str,
        dev_file_path: str,
        model: Lazy[TransformerModel],
        tokenizer: Lazy[TransformerTokenizer],
        dataset_reader: Lazy[DatasetReader],
        data_collator: Lazy[TransformerDataCollator],
        optimizer: Lazy[Optimizer],
        compute_metrics: Optional[Lazy[TransformerMetric]] = None,
        no_grad: List[str] = None,
        args: TransformerTrainingArguments = None,
        callbacks: Optional[List[TrainerCallback]] = None,
    ) -> "TransformerTrainer":

        #  To find the registrable components from the user-defined packages
        import_plugins()

        tokenizer_ = tokenizer.construct(
            pretrained_model_name_or_path=pretrained_model_name_or_path
        )
        model_ = model.construct(
            pretrained_model_name_or_path=pretrained_model_name_or_path
        )
        cls._resize_token_embeddings(model=model_, tokenizer=tokenizer_)
        cls.mark_params_with_no_grads(model_, no_grad)
        params_with_grad = [
            [n, p] for n, p in model_.named_parameters() if p.requires_grad
        ]
        optimizer_ = optimizer.construct(model_parameters=params_with_grad)

        data_collator_ = data_collator.construct(
            tokenizer=tokenizer_, model_input_keys=model_.forward_params
        )
        compute_metrics_ = cls._create_compute_metrics(
            compute_metrics, data_collator_
        )
        dataset_reader_ = dataset_reader.construct(tokenizer=tokenizer_)
        train_dataset_ = dataset_reader_.read(train_file_path)
        eval_dataset_ = dataset_reader_.read(dev_file_path)

        return cls(
            model=model_,
            args=args,
            data_collator=data_collator_,
            train_dataset=train_dataset_,
            eval_dataset=eval_dataset_,
            tokenizer=tokenizer_,
            compute_metrics=compute_metrics_,
            callbacks=callbacks,
            optimizers=(optimizer_, None),
        )

    @classmethod
    def _create_compute_metrics(
        cls,
        compute_metrics: Optional[Lazy[TransformerMetric]],
        data_collator: TransformerDataCollator,
    ) -> Optional[TransformerMetric]:
        if compute_metrics is None:
            return None
        label_list = getattr(data_collator, "label_list", None)
        if label_list is None:
            raise ValueError(
                f"The data collator {str(type(data_collator))}"
                + " must implement the label_list attribute."
            )
        return compute_metrics.construct(label_list=data_collator.label_list)

    @classmethod
    def mark_params_with_no_grads(cls, model_, no_grad):
        if no_grad:
            for name, parameter in model_.named_parameters():
                if any(re.search(regex, name) for regex in no_grad):
                    parameter.requires_grad_(False)

    @classmethod
    def _resize_token_embeddings(
        cls, model: PreTrainedModel, tokenizer: TransformerTokenizer
    ):
        """
        Update the token embedding layer of the model to accommodate
        for the special tokens added to the tokenizer
        """
        if tokenizer.num_added_special_tokens > 0:
            model.resize_token_embeddings(new_num_tokens=tokenizer.num_tokens)


TransformerTrainer.register("default", constructor="from_partial_objects")(
    TransformerTrainer
)
