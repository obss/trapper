from abc import ABC, abstractmethod

from trapper.common import Registrable
from trapper.data.data_processors.data_processor import IndexedInstance
from trapper.data.tokenizers.tokenizer import TransformerTokenizer


class DataAdapter(ABC, Registrable):
    """
    This callable class is responsible from converting the data instances that
    are already tokenized and indexed into a format suitable for feeding into a
    transformer model. Typically, it receives its inputs from a `DataProcessor`
    and adapts the input fields by renaming them to the names accepted by the
    models e.g. "input_ids", "token_type_ids" etc. Moreover, it also handles
    the insertion of the special tokens signaling the beginning or ending of a
    sequence such as `[CLS]`, `[SEP]` etc. You need to implement the `__call__`
    method suitable for your task when you subclass it. See
    `DataAdapterForQuestionAnswering` for an example.

    Args:
        tokenizer (): Required to access the ids of special tokens
    """

    def __init__(self, tokenizer: TransformerTokenizer):
        self._tokenizer = tokenizer

    @abstractmethod
    def __call__(self, instance: IndexedInstance, split: str) -> IndexedInstance:
        """
        Takes a raw `IndexedInstance`, performs some processing on it, and returns
        an `IndexedInstance` again. Look at
        `DataAdapterForQuestionAnswering.__call__` for an example.

        Args:
            instance ():
            split (): Required to determine validation samples for evaluation
        """
        raise NotImplementedError
