import inspect
import logging
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Union

from torch.utils.data import Dataset

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


class IndexedDataset(Dataset):
    def __init__(self, instances: List[IndexedInstance]):
        self.instances = instances

    def __getitem__(self, index) -> IndexedInstance:
        return self.instances[index]

    def __len__(self):
        return len(self.instances)

    def __iter__(self) -> Iterator[IndexedInstance]:
        yield from self.instances


class TransformerDataProcessor(Registrable, metaclass=ABCMeta):
    """
    This class is used to read a dataset file and return a collection of
    `IndexedDataset`s. The abstract `_read` and `text_to_instance` must be
    implemented in the subclasses. Typically, `_read` method calls
    `text_to_instance` with raw data as input to generate `IndexedInstance`s.
    Some methods that are commonly used are implemented here for convenience.

    Child classes have to set the following class variables:
        - NUM_EXTRA_SPECIAL_TOKENS_IN_SEQUENCE : The total number of extra special
            tokens (not unique) in the input ids.

    Args:
        tokenizer ():

        From basicdatasetreader:
        This class is used to read a dataset file and return a collection of
    `IndexedDataset`s. The abstract `_read` and `text_to_instance` must be
    implemented in the subclasses. Typically, `_read` method calls
    `text_to_instance` with raw data as input to generate `IndexedInstance`s.
    Some methods that are commonly used are implemented here for convenience
    """

    NUM_EXTRA_SPECIAL_TOKENS_IN_SEQUENCE = None

    def __init__(self, tokenizer: TransformerTokenizer):
        self._tokenizer = tokenizer

    @property
    def tokenizer(self):
        return self._tokenizer

    @abstractmethod
    def text_to_instance(self, *inputs) -> IndexedInstance:
        """
        Takes unpacked, raw input and converts them to an `IndexedInstance`.
        Typically, called by the `_read` method. An important suggestion while
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
    def process(self, instance_dict: Dict[str, Any]) -> Optional[IndexedInstance]:
        """
        Processes an instance dict coming form a `datasets.Dataset`. Returns the
        indexed instance if the input is successfully tokenized, indexed and
        arranged. Otherwise, returns None.

        Args:
            instance_dict ():
        """
        raise NotImplementedError

    @classmethod
    def _total_seq_len(cls, *sequences):
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