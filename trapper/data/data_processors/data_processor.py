import logging
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional, Union

from trapper.common import Registrable
from trapper.common.constants import PositionTuple
from trapper.data.tokenizers.tokenizer import TransformerTokenizer

logger = logging.getLogger(__file__)

IndexedInstance = Dict[
    str, Union[int, List[int], PositionTuple, List[PositionTuple]]
]


class ImproperDataInstanceError(Exception):
    """Raised when the input is not suitable for use because its fields
    are not compatible e.g. because of size mismatch or pre-processing
     artifacts"""


class DataProcessor(Registrable, metaclass=ABCMeta):
    """
    A callable class used for converting a raw instance dict from `datasets.Dataset`
    to `IndexedInstance`. Typically, used as the first processing step after the
    raw data is read to extract the task-related fields from the raw data.
    The abstract `text_to_instance` and `__call__` must be implemented in the
    subclasses. Typically, the `__call__` method calls `text_to_instance` with
    raw data as input to generate an `IndexedInstance`. Some methods that are
    commonly used are implemented here for convenience.

    Child classes may need to set the following class variables:
        - NUM_EXTRA_SPECIAL_TOKENS_IN_SEQUENCE : The total number of extra special
            tokens (do not have to be unique) in the input ids.

    Args:
        tokenizer ():
    """

    NUM_EXTRA_SPECIAL_TOKENS_IN_SEQUENCE = 0

    def __init__(self, tokenizer: TransformerTokenizer):
        self._tokenizer = tokenizer

    @property
    def tokenizer(self):
        return self._tokenizer

    @abstractmethod
    def text_to_instance(self, *inputs) -> IndexedInstance:
        """
        Takes unpacked, raw input and converts them to an `IndexedInstance`.
        Typically, invoked by the `__call__` method. An important suggestion while
        implementing this method in your custom subclass is to put the label input
        at the end as an optional parameter whose default value is `None`. By this
        way, you can reuse this method while reading a single instance during the
        inference (e.g. through a `pipeline` or an inference script) when the data
        does not have the label information.

        Args:
            *inputs (): raw input
        """
        raise NotImplementedError

    @abstractmethod
    def __call__(self, instance_dict: Dict[str, Any]) -> IndexedInstance:
        """
        Processes an instance dict taken from a `datasets.Dataset`. Typically,
        extracts the task-related fields and pass them to `text_to_instance` method.
        Returns an`IndexedInstance` with proper keys if the input is successfully
        tokenized, indexed and arranged, otherwise returns a dummy
        `IndexedInstance` with "filter_out"=True and the remaining keys are
        associated with empty values suitable with the expected types.

        Args:
            instance_dict ():
        """
        raise NotImplementedError

    @classmethod
    def _total_seq_len(cls, *sequences):
        """
        Computes the total number of tokens in an iterable of sequences by taking
        the special tokens in the combined sequence into account as well.

        Args:
            *sequences ():
        """
        total_seq_len = sum(len(seq) for seq in sequences)
        total_seq_len += cls.NUM_EXTRA_SPECIAL_TOKENS_IN_SEQUENCE
        return total_seq_len

    def _chop_excess_tokens(self, tokens: List, max_len: int):
        """
        Utilizes a  heuristic of chopping off the excess tokens in-place
         from the end
        """
        excess = max_len - self._tokenizer.model_max_sequence_length
        del tokens[-1 * excess :]
