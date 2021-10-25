import logging
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional, Union

from transformers import PreTrainedTokenizerBase

from trapper.common import Registrable
from trapper.common.constants import PositionDict
from trapper.data.label_mapper import LabelMapper
from trapper.data.tokenizers import TokenizerWrapper

logger = logging.getLogger(__file__)

IndexedInstance = Dict[str, Union[int, List[int], PositionDict, List[PositionDict]]]


class ImproperDataInstanceError(Exception):
    """Raised when the input is not suitable for use because its fields
    are not compatible e.g. because of size mismatch or pre-processing
     artifacts"""


class DataProcessor(Registrable, metaclass=ABCMeta):
    """
    A callable class used for converting a raw instance dict from `datasets.Dataset`
    to `IndexedInstance`. Typically, used as the first processing step after the
    raw data is read to extract the task-related fields from the raw data.
    The abstract `text_to_instance` and `process` methods must be implemented in the
    subclasses. Typically, the `process` method calls `text_to_instance` with
    raw data as input to generate an `IndexedInstance`. Some methods that are
    commonly used are implemented here for convenience.

    Child classes may need to set the following class variables:
        - NUM_EXTRA_SPECIAL_TOKENS_IN_SEQUENCE : The total number of extra special
            tokens (do not have to be unique) in the input ids.

    Args:
        tokenizer_wrapper ():
        model_max_sequence_length (): The maximum length of the processed
            sequence. Actually, the maximum sequence will be the minimum of this
            value and the `model_max_length` value of the tokenizer.
        label_mapper (): Only used in some tasks that require mapping between
            categorical labels and integer ids such as token classification.
    """

    NUM_EXTRA_SPECIAL_TOKENS_IN_SEQUENCE = 0

    def __init__(
        self,
        tokenizer_wrapper: TokenizerWrapper,
        model_max_sequence_length: Optional[int] = None,
        label_mapper: Optional[LabelMapper] = None,
    ):
        self._tokenizer: PreTrainedTokenizerBase = tokenizer_wrapper.tokenizer
        self._model_max_sequence_length = self._find_model_max_seq_length(
            model_max_sequence_length
        )
        self._label_mapper = label_mapper

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model_max_sequence_length(self) -> int:
        return self._model_max_sequence_length

    def _find_model_max_seq_length(
        self,
        provided_model_max_sequence_length: int = None,
    ) -> int:
        model_max_length = self.tokenizer.model_max_length
        if provided_model_max_sequence_length is None:
            return model_max_length
        return min(model_max_length, provided_model_max_sequence_length)

    @abstractmethod
    def text_to_instance(self, *inputs) -> IndexedInstance:
        """
        Takes unpacked, raw input and converts them to an `IndexedInstance`.
        Typically, invoked by the `process` method to tokenize the raw instance
        and arrange the token indices. An important suggestion while
        implementing this method in your custom subclass is to put the label input
        at the end as an optional parameter whose default value is `None`. By this
        way, you can reuse this method while reading a single instance during the
        inference (e.g. through a `pipeline` or an inference script) when the data
        does not have the label information.

        Args:
            *inputs (): raw input
        """
        raise NotImplementedError

    def __call__(self, instance_dict: Dict[str, Any]) -> IndexedInstance:
        """
        Processes an instance dict taken from a `datasets.Dataset` by calling the
        `process` method on it. Moreover, checks the returned instance for validity.
        If `__discard_sample` key is not found in the returned instance, it is
        added with the value of `True` to mark the returned instance as valid.
        Then, simply returns the indexed instance.

        Args:
            instance_dict (): The raw sample

        Returns:
            The processed instance in `IndexedInstance` format
        """
        indexed_instance = self.process(instance_dict)
        indexed_instance["__discard_sample"] = indexed_instance.get(
            "__discard_sample", False
        )
        return indexed_instance

    def process(self, instance_dict: Dict[str, Any]) -> IndexedInstance:
        """
        Processes an instance dict which is typically taken from a
        `datasets.Dataset`. Its intended use is to extract the task-related fields
        and pass them to `text_to_instance` method. Returns an`IndexedInstance`
        with proper keys if the input is successfully tokenized, indexed and
        arranged, otherwise returns a dummy`IndexedInstance` with
        "__discard_sample" key set to `True` and the remaining keys are
        associated with empty values suitable to the expected types.

        Args:
            instance_dict (): The raw sample

        Returns:
            The processed instance in `IndexedInstance` format
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

    def _chop_excess_tokens(self, sequence: List, total_len: int):
        """
        Chops the excess tokens in a sequence from the right side. `total_len`
        is the current total length computed in some way before calling this
        function. If the caller deals with a single sequence,
        `total_len == len(sequence)` should hold. However, if the caller will
        append two or more sequences next to each other, `total_len` should be the
        current total length of the sequences, and the `sequence` argument should be
        the first one. This is useful in contextual tasks such as question
        answering where we typically append the context and question next to each
        other. In this case, the context should be supplied as the `sequence`
        argument so that we chop toward the end of it (right side) without chopping
        the question.

        Args:
            sequence (): the input sequence. If the caller deals with multiple
                sequences, the first (leftmost) one should be passed
            total_len (): total length before calling this function
        """
        excess = total_len - self.model_max_sequence_length
        del sequence[-1 * excess :]
