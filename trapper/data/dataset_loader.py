import inspect
import logging
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Union

import datasets
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

from trapper.common import Registrable
from trapper.data.data_processors import IndexedDataset, IndexedInstance
from trapper.data.data_processors.data_processor import TransformerDataProcessor

logger = logging.getLogger(__file__)


class DatasetLoader(Registrable):
    """
    This class is used to read a dataset from the `datasets` library and return
    a collection of `IndexedDataset`s. Its first argument is a
    `TransformerDataProcessor` which is required for tokenization and handling
    the special tokens during the pre-processing. The remaining arguments are
    directly transferred to the `datasets.load_dataset` function and use for
    loading the raw dataset from the `datasets` library. See
    :py:func:`datasets.load_dataset` for the details of these parameters and how
    the dataset is loaded.

    Args:
        data_processor ():
        path ():
        name ():
        data_dir ():
        data_files ():
        split ():
        cache_dir ():
        features ():
        download_config ():
        download_mode ():
        ignore_verifications ():
        keep_in_memory ():
        save_infos ():
        script_version ():
        use_auth_token ():
        task ():
        streaming ():
        **config_kwargs ():
    """

    default_implementation = "default"

    def __init__(
        self,
        data_processor: TransformerDataProcessor,
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
        self._data_processor = data_processor
        locals_ = locals()
        self._dataset = load_dataset(
            *[locals_[arg] for arg in inspect.getfullargspec(load_dataset).args],
            **config_kwargs,
        )

    @property
    def data_processor(self):
        return self._data_processor

    @property
    def split_names(self):
        return tuple(
            s for s in ("train", "validation", "test") if s in self._dataset
        )

    def load(self, split_name: Union[Path, str]) -> IndexedDataset:
        """
        Reads the dataset for the specified split.

        Args:
            split_name (): one of "train", "validation" or "test"

        Returns:
            an `IndexedDataset` that can be passed to `TransformerTrainer`
        """
        # TODO: Check if `ensure_list` is really necessary?
        instances = ensure_list(self._load(self._get_raw_data(split_name)))
        return IndexedDataset(instances)

    def _load(self, split: datasets.Dataset) -> Iterable[IndexedInstance]:
        """
        Returns an `Iterable` of `IndexedInstance`s from a dataset split.

        Args:
            split (): The train, validation or test split of a dataset

        Returns:
            an iterable of `IndexedInstance`s
        """
        for instance_dict in split:
            indexed_instance = self.data_processor(instance_dict)
            if indexed_instance is not None:
                yield indexed_instance

    def _get_raw_data(self, split_name: Union[Path, str]):
        self._check_split_name(split_name)
        return self._dataset[split_name]

    def _check_split_name(self, split_name):
        if split_name not in self.split_names:
            raise ValueError(
                f"split: {split_name} not in available splits for the "
                f"dataset ({self.split_names})"
            )


DatasetLoader.register("default")(DatasetLoader)
