import logging
import os
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import (
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Union,
)

import datasets
import torch
from allennlp.common.util import ensure_list
from datasets import (
    DownloadConfig,
    Features,
    GenerateMode,
    Split,
    Version,
    load_dataset,
)
from datasets.tasks import TaskTemplate
from torch.utils.data import Dataset

from trapper.common import Registrable
from trapper.common.constants import PositionTuple
from trapper.common.utils import append_callable_docstr, append_parent_docstr
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


class DatasetReader(Registrable, metaclass=ABCMeta):
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


@append_callable_docstr(callable_=load_dataset)
class HuggingFaceDatasetReader(DatasetReader, metaclass=ABCMeta):
    """
    This class is used to read a dataset from the `datasets` library and return a
    collection of `IndexedDataset`s. It's first argument is the tokenizer which
    is needed for tokenization during pre-processing. The remaining arguments are
    used for data loading and directly transferred to the `datasets.load_dataset`
    function. Therefore, the remaining docstring is automatically copied from that
    function.
    """

    def __init__(
        self,
        tokenizer: TransformerTokenizer,
        path: str,
        name: Optional[str] = None,
        data_dir: Optional[str] = None,
        data_files: Optional[
            Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]
        ] = None,
        split: Optional[Union[str, Split]] = None,
        cache_dir: Optional[str] = None,
        features: Optional[Features] = None,
        download_config: Optional[DownloadConfig] = None,
        download_mode: Optional[GenerateMode] = None,
        ignore_verifications: bool = False,
        keep_in_memory: Optional[bool] = None,
        save_infos: bool = False,
        script_version: Optional[Union[str, Version]] = None,
        use_auth_token: Optional[Union[bool, str]] = None,
        task: Optional[Union[str, TaskTemplate]] = None,
        streaming: bool = False,
        **config_kwargs,
    ):
        super().__init__(tokenizer)
        self._dataset = load_dataset(
            path=path,
            **{
                k: v
                for k, v in locals().items()
                if k not in ("self", "tokenizer", "path")
            },
        )

    @property
    def split_names(self):
        return tuple(
            s for s in ("train", "validation", "test") if s in self._dataset
        )

    @abstractmethod
    def _read(self, split: datasets.Dataset) -> Iterable[IndexedInstance]:
        """
        Returns an `Iterable` of `IndexedInstance`s from the dataset split.

        Args:
            split (): A train, validation or test split of a dataset

        Returns:
            an iterable of IndexedInstance's read from the dataset split
        """
        raise NotImplementedError

    def read(self, split_name: Union[Path, str]) -> IndexedDataset:
        """
        Reads the dataset for the specified split.

        Args:
            split_name (): one of "train", "validation" or "test"

        Returns:
            an `IndexedDataset` that can be passed to `TransformerTrainer`
        """
        if split_name not in self.split_names:
            raise ValueError(
                f"split: {split_name} not in available splits for the "
                f"dataset ({self.split_names})"
            )

        instances = ensure_list(self._read(self._dataset[split_name]))
        return IndexedDataset(instances)


class BasicDatasetReader(DatasetReader):
    """
    This class is used to read a dataset file and return a collection of
    `IndexedDataset`s. The abstract `_read` and `text_to_instance` must be
    implemented in the subclasses. Typically, `_read` method calls
    `text_to_instance` with raw data as input to generate `IndexedInstance`s.
    Some methods that are commonly used are implemented here for convenience.

    Args:
        tokenizer ():
        apply_cache ():
        cache_file_prefix ():
        cache_directory ():
    """

    def __init__(
        self,
        tokenizer: TransformerTokenizer,
        apply_cache: bool = False,
        cache_file_prefix: Optional[str] = None,
        cache_directory: Optional[Union[str, Path]] = None,
    ):
        super().__init__(tokenizer)
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

    @abstractmethod
    def _read(self, file_path: Union[Path, str]) -> Iterable[IndexedInstance]:
        """
        Returns an `Iterable` of `IndexedInstance`s that was read from the input
        file path.
        Args:
            file_path (): input file path

        Returns:
            an iterable of IndexedInstance's read from the dataset.
        """
        raise NotImplementedError

    def read(self, file_path: Union[Path, str]) -> IndexedDataset:
        """
        Reads the dataset from the file_path.
        Args:
            file_path ():

        Returns:
            an `IndexedDataset` that can be passed to `TransformerTrainer`
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

    @staticmethod
    def _read_from_cache(cache_path: Path) -> IndexedDataset:
        instances = torch.load(cache_path)
        return IndexedDataset(instances)
