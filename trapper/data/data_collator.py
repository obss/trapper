from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase

from trapper.common import Registrable
from trapper.common.constants import IGNORED_LABEL_ID
from trapper.data.data_processors.data_processor import IndexedInstance
from trapper.data.tokenizers import TokenizerWrapper

InputBatch = Dict[str, List[Union[int, List[int]]]]
InputBatchTensor = Dict[str, Tensor]


class DataCollator(Registrable):
    """
    This class takes a batch of `IndexedInstance`s, typically generated using a
    dataset reader, and returns an `InputBatchTensor`. It is responsible from
    padding the required instances, collect them into a batch and convert it into a
    `InputBatchTensor`. Moreover, it also creates the attention_mask and inserts
    it into the batch if required. It is used as a callable just like the
    `DataCollator` in the `transformers` library.

    Args:
        tokenizer ():
        model_forward_params ():
    """

    default_implementation = "default"

    def __init__(
        self,
        tokenizer_wrapper: TokenizerWrapper,
        model_forward_params: Tuple[str, ...],
    ):
        self._tokenizer: PreTrainedTokenizerBase = tokenizer_wrapper.tokenizer
        self._model_forward_params: Tuple[str, ...] = model_forward_params

    def __call__(
        self,
        instances: Iterable[IndexedInstance],
        should_eliminate_model_incompatible_keys: bool = True,
    ) -> InputBatchTensor:
        """Prepare the dataset for training and evaluation"""
        batch = self.build_model_inputs(
            instances,
            should_eliminate_model_incompatible_keys=should_eliminate_model_incompatible_keys,
        )
        self.pad(batch)
        return self._convert_to_tensor(batch)

    def build_model_inputs(
        self,
        instances: Iterable[IndexedInstance],
        return_attention_mask: Optional[bool] = None,
        should_eliminate_model_incompatible_keys: bool = True,
    ) -> InputBatch:
        return_attention_mask = (
            return_attention_mask or "attention_mask" in self._model_forward_params
        )
        batch = self._create_empty_batch()
        for instance in instances:
            if should_eliminate_model_incompatible_keys:
                self._eliminate_model_incompatible_keys(instance)
            if return_attention_mask:
                self._add_attention_mask(instance)
            self._add_instance(batch, instance)

        self._eliminate_empty_inputs(batch)
        return batch

    @staticmethod
    def _create_empty_batch() -> InputBatch:
        return defaultdict(list)

    def _eliminate_model_incompatible_keys(self, instance: IndexedInstance):
        incompatible_keys = [
            key for key in instance.keys() if key not in self._model_forward_params
        ]
        for key in incompatible_keys:
            del instance[key]

    @staticmethod
    def _add_attention_mask(
        instance: IndexedInstance,
    ):
        if "attention_mask" not in instance:
            instance["attention_mask"] = [1] * len(instance["input_ids"])

    @staticmethod
    def _add_instance(batch: InputBatch, instance: IndexedInstance):
        for field_name, encodings in instance.items():
            batch[field_name].append(encodings)

    @staticmethod
    def _eliminate_empty_inputs(batch: InputBatch):
        incompatible_keys = [key for key, val in batch.items() if len(val) == 0]
        for key in incompatible_keys:
            del batch[key]

    def pad(
        self,
        batch: InputBatch,
        max_length: int = None,
        padding_side: str = "right",
    ):
        for feature_key, feature_values in batch.items():
            if not isinstance(feature_values[0], int):  # use str keys
                max_seq_len = max(len(ids) for ids in feature_values)
                padded_len = max_seq_len if max_length is None else max_length
                pad_id = self._pad_id(feature_key)
                self._pad_encodings(
                    encodings=feature_values,
                    pad_id=pad_id,
                    padded_len=padded_len,
                    padding_side=padding_side,
                )

    def _pad_id(self, padded_field: str) -> int:
        if padded_field == "input_ids":
            pad_id = self._tokenizer.pad_token_id
        elif padded_field == "token_type_ids":
            pad_id = self._tokenizer.pad_token_type_id
        elif padded_field == "labels":
            pad_id = IGNORED_LABEL_ID
        elif padded_field == "attention_mask":
            pad_id = 0
        elif padded_field == "special_tokens_mask":
            pad_id = 1
        else:
            raise ValueError(f"{padded_field} is not a valid field for padding")
        return pad_id

    @staticmethod
    def _pad_encodings(
        encodings: List[List[int]],
        pad_id: int,
        padded_len: int,
        padding_side: str = "right",
    ):
        for i, encoded_inputs in enumerate(encodings):
            difference = padded_len - len(encoded_inputs)
            if difference == 0:
                continue
            pad_values = [pad_id] * difference
            if padding_side == "right":
                encoded_inputs.extend(pad_values)
            elif padding_side == "left":
                encodings[i] = pad_values + encoded_inputs

    @staticmethod
    def _convert_to_tensor(batch: InputBatch) -> InputBatchTensor:
        return {
            input_key: torch.tensor(encodings)
            for input_key, encodings in batch.items()
        }


DataCollator.register("default")(DataCollator)
