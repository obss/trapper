import inspect
import logging
from typing import Mapping, Optional, Sequence, Union

from datasets import (
    Dataset,
    DatasetDict,
    DownloadConfig,
    Features,
    GenerateMode,
    IterableDataset,
    IterableDatasetDict,
    Split,
    Version,
    load_dataset,
)
from datasets.tasks import TaskTemplate

from trapper.common import Registrable

logger = logging.getLogger(__file__)

TrapperDataset = Union[Dataset, IterableDataset]
TrapperDatasetDict = Union[DatasetDict, IterableDatasetDict]


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
        self._init_params = locals()
        self._init_config_kwargs = config_kwargs

    def read(
        self, split: Optional[Union[str, Split]] = None
    ) -> Union[TrapperDataset, TrapperDatasetDict]:
        """
        Returns the specified split of the dataset.

        Args:
            split (): Which split of the data to load. If None, will return a
                `dict` with all splits e.g. "train" and "validation". If specified,
                will return a single Dataset
        """
        init_params = self._init_params
        init_params["split"] = split
        return load_dataset(
            *[
                init_params[arg]
                for arg in inspect.getfullargspec(load_dataset).args
            ],
            **self._init_config_kwargs,
        )


DatasetReader.register("default")(DatasetReader)
