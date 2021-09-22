from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union

from torch import Tensor

from trapper.common import Registrable
from trapper.data.data_processors.data_processor import IndexedInstance
from trapper.data.tokenizers.tokenizer import TransformerTokenizer

InputBatch = Dict[str, List[Union[int, List[int]]]]
InputBatchTensor = Dict[str, Tensor]


class DataAdapter(ABC, Registrable):
    def __init__(
            self,
            tokenizer: TransformerTokenizer,
            model_input_keys: Tuple[str, ...],
    ):
        self._tokenizer = tokenizer
        self._model_input_keys: Tuple[str, ...] = model_input_keys

    # def __call__(self, instances: IndexedInstance) -> IndexedInstance:
    #     # Replaces instance = self._build_input_fields(instance)
    #     raise NotImplementedError

    @abstractmethod
    def _build_input_fields(self, instance: IndexedInstance) -> IndexedInstance:
        """
        Takes a raw `IndexedInstance`, performs some processing on it,
        and returns an `IndexedInstance` again. Look at
        `DataCollatorForQuestionAnswering._build_input_fields` for an example.
        Args:
            instance ():
        """
        raise NotImplementedError

    @staticmethod
    def _extend_token_ids(
            instance: IndexedInstance, token_type_id: int, input_ids: List[int]
    ):
        instance["input_ids"].extend(input_ids)
        token_type_ids = [token_type_id] * len(input_ids)
        instance["token_type_ids"].extend(token_type_ids)
