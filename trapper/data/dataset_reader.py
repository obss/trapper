import inspect
import logging
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

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

logger = logging.getLogger(__file__)


class DatasetReader(Registrable):
    """
    This class is used for reading a raw dataset from the `datasets` library. The
    dataset is read and stored during `DatasetReader` object instantiation so
    that the available splits (train, validation etc) can be queried later without
    reading it again. The constructor arguments are directly transferred to the
    `datasets.load_dataset` function. See `datasets.load_dataset`_ for the
    details of these parameters and how the dataset is loaded.

    .. _datasets.load_dataset: https://github.com/huggingface/datasets/blob/b057846bcc1ccfe0fda4d8d42f190d146a70ee64/src/datasets/load.py#L996

    Args:
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
        locals_ = locals()
        self._dataset = load_dataset(
            *[locals_[arg] for arg in inspect.getfullargspec(load_dataset).args],
            **config_kwargs,
        )
        self._split_names = tuple(s for s in ("train", "validation", "test")
                                  if s in self._dataset)

    @property
    def split_names(self):
        return self._split_names

    def get_dataset(self, split_name: Union[Path, str]):
        """
        Returns the specified split of the dataset.

        Args:
            split_name (): one of "train", "validation" or "test"
        """
        self._check_split_name(split_name)
        return self._dataset[split_name]

    def _check_split_name(self, split_name):
        if split_name not in self.split_names:
            raise ValueError(
                f"split: {split_name} not in available splits for the "
                f"dataset ({self.split_names})"
            )


DatasetReader.register("default")(DatasetReader)
