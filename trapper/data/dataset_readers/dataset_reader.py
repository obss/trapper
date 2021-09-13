import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Union

import torch
from allennlp.common.util import ensure_list
from torch.utils.data import Dataset

from trapper.common import Registrable
from trapper.common.constants import PositionTuple
from trapper.data.tokenizers.tokenizer import TransformerTokenizer

logger = logging.getLogger(__file__)


class ImproperDataInstanceError(Exception):
    """Raised when the input is not suitable for use because its fields
    are not compatible e.g. because of size mismatch or pre-processing
     artifacts"""


IndexedInstance = Dict[
    str, Union[int, List[int], PositionTuple, List[PositionTuple]]
]


class IndexedDataset(Dataset):
    def __init__(self, instances: List[IndexedInstance]):
        self.instances = instances

    def __getitem__(self, index) -> IndexedInstance:
        return self.instances[index]

    def __len__(self):
        return len(self.instances)

    def __iter__(self) -> Iterator[IndexedInstance]:
        yield from self.instances


class DatasetReader(ABC, Registrable):
    """
    This class is used to read a dataset file and return a collection of
    `IndexedDataset`s. The abstract `_read` and `text_to_instance` must be
    implemented in the subclasses. Typically, `_read` method calls
    `text_to_instance` with raw data as input to generate `IndexedInstance`s.
    Some methods that are commonly used are implemented here for convenience.

    Child classes have to set the following class variables:
        - NUM_EXTRA_SPECIAL_TOKENS_IN_SEQUENCE : The total number of extra special tokens (not unique)
                                                in the input ids.

    Args:
        tokenizer ():
        apply_cache ():
        cache_file_prefix ():
        cache_directory ():
    """

    NUM_EXTRA_SPECIAL_TOKENS_IN_SEQUENCE = None

    def __init__(
        self,
        tokenizer: TransformerTokenizer,
        apply_cache: bool = False,
        cache_file_prefix: Optional[str] = None,
        cache_directory: Optional[Union[str, Path]] = None,
    ):
        self._tokenizer = tokenizer
        self._apply_cache = apply_cache
        if apply_cache:
            if cache_file_prefix is None:
                cache_file_prefix = self._create_cache_prefix()
            self._cache_prefix = cache_file_prefix
            if cache_directory is None:
                raise ValueError("Provide a cache directory!")
            self._cache_directory = Path(cache_directory)

    def _create_cache_prefix(self):
        cache_file_prefix = ""
        for obj in (self, self._tokenizer, self._tokenizer._tokenizer):
            cache_file_prefix += f"{type(obj).__name__}_"
        return cache_file_prefix

    @property
    def tokenizer(self):
        return self._tokenizer

    @abstractmethod
    def _read(self, file_path: Union[Path, str]) -> Iterable[IndexedInstance]:
        """
        Returns an `Iterable` of `IndexedInstance`s that was read from the input
        file path.
        Args:
            file_path (): input file path

        Returns:
            an iterable of IndexedInstance's read from the dataset
        """
        raise NotImplementedError

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

    def read(self, file_path: Union[Path, str]) -> IndexedDataset:
        """
        Reads the dataset from the file_path and generates an `IndexedDataset`
        that can be passed to `TransformerTrainer`.
        Args:
            file_path ():

        Returns:

        """
        if not os.path.isfile(file_path):
            raise ValueError(f"{file_path} is not a valid file path")
        file_path = Path(file_path)

        cache_file_path = None
        if self._apply_cache:
            cache_file_path = self._get_cache_path(file_path)
            if os.path.isfile(cache_file_path):
                logger.info(
                    "Load tokenized dataset from cache found at %s",
                    cache_file_path,
                )
                return self._read_from_cache(cache_file_path)

        instances = ensure_list(self._read(file_path))
        if self._apply_cache:
            self._cache_dataset(cache_file_path, instances)
        return IndexedDataset(instances)

    @staticmethod
    def _cache_dataset(cache_file_path: Path, dataset: List[IndexedInstance]):
        os.makedirs(cache_file_path.parent, exist_ok=True)
        torch.save(dataset, cache_file_path)
        logger.info("Dataset cached at %s", cache_file_path)

    def _get_cache_path(self, file_path: Union[Path, str]) -> Path:
        cache_file = (
            f"dataset_cache_{self._cache_prefix}_"
            + f"srcdir{file_path}".replace("/", "_").replace(".", "_")
        )
        cache_file_path = self._cache_directory / "datasets" / cache_file
        return cache_file_path

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

    @staticmethod
    def _read_from_cache(cache_path: Path) -> IndexedDataset:
        instances = torch.load(cache_path)
        return IndexedDataset(instances)
