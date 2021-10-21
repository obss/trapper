# Copyright 2021 Open Business Software Solutions, the HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
    This class is used for reading a raw dataset from the `datasets` library.
    The constructor arguments are stored so that they can be used from the `read`
    method to read the dataset using the `datasets.load_dataset` function . Notice
    that the constructor does not have `split` parameter. Instead, the split is
    specified inside the `read` method as the only parameter. See
    `datasets.load_dataset`_ for the details of how the dataset is loaded.

    .. _datasets.load_dataset: https://github.com/huggingface/datasets/blob/ec824222c227ea5c9c75568d6f357a819599a6c7/src/datasets/load.py#L1472

    Args:

        path (:obj:`str`): Path or name of the dataset.
            Depending on ``path``, the dataset builder that is returned id either generic dataset builder (csv, json, text etc.) or a dataset builder defined defined a dataset script (a python file).

            For local datasets:

            - if ``path`` is a local directory (but doesn't contain a dataset script)
              -> load a generic dataset builder (csv, json, text etc.) based on the content of the directory
              e.g. ``'./path/to/directory/with/my/csv/data'``.
            - if ``path`` is a local dataset script or a directory containing a local dataset script (if the script has the same name as the directory):
              -> load the dataset builder from the dataset script
              e.g. ``'./dataset/squad'`` or ``'./dataset/squad/squad.py'``.

            For datasets on the Hugging Face Hub (list all available datasets and ids with ``datasets.list_datasets()``)

            - if ``path`` is a canonical dataset on the HF Hub (ex: `glue`, `squad`)
              -> load the dataset builder from the dataset script in the github repository at huggingface/datasets
              e.g. ``'squad'`` or ``'glue'``.
            - if ``path`` is a dataset repository on the HF hub (without a dataset script)
              -> load a generic dataset builder (csv, text etc.) based on the content of the repository
              e.g. ``'username/dataset_name'``, a dataset repository on the HF hub containing your data files.
            - if ``path`` is a dataset repository on the HF hub with a dataset script (if the script has the same name as the directory)
              -> load the dataset builder from the dataset script in the dataset repository
              e.g. ``'username/dataset_name'``, a dataset repository on the HF hub containing a dataset script `'dataset_name.py'`.

        name (:obj:`str`, optional): Defining the name of the dataset configuration.
        data_dir (:obj:`str`, optional): Defining the data_dir of the dataset configuration.
        data_files (:obj:`str` or :obj:`Sequence` or :obj:`Mapping`, optional): Path(s) to source data file(s).
        split (:class:`Split` or :obj:`str`): Which split of the data to load.
            If None, will return a `dict` with all splits (typically `datasets.Split.TRAIN` and `datasets.Split.TEST`).
            If given, will return a single Dataset.
            Splits can be combined and specified like in tensorflow-datasets.
        cache_dir (:obj:`str`, optional): Directory to read/write data. Defaults to "~/.cache/huggingface/datasets".
        features (:class:`Features`, optional): Set the features type to use for this dataset.
        download_config (:class:`~utils.DownloadConfig`, optional): Specific download configuration parameters.
        download_mode (:class:`GenerateMode`, default ``REUSE_DATASET_IF_EXISTS``): Download/generate mode.
        ignore_verifications (:obj:`bool`, default ``False``): Ignore the verifications of the downloaded/processed dataset information (checksums/size/splits/...).
        keep_in_memory (:obj:`bool`, default ``None``): Whether to copy the dataset in-memory. If `None`, the dataset
            will not be copied in-memory unless explicitly enabled by setting `datasets.config.IN_MEMORY_MAX_SIZE` to
            nonzero. See more details in the :ref:`load_dataset_enhancing_performance` section.
        save_infos (:obj:`bool`, default ``False``): Save the dataset information (checksums/size/splits/...).
        revision (:class:`~utils.Version` or :obj:`str`, optional): Version of the dataset script to load:

            - For canonical datasets in the `huggingface/datasets` library like "squad", the default version of the module is the local version of the lib.
              You can specify a different version from your local version of the lib (e.g. "master" or "1.2.0") but it might cause compatibility issues.
            - For community provided datasets like "lhoestq/squad" that have their own git repository on the Datasets Hub, the default version "main" corresponds to the "main" branch.
              You can specify a different version that the default "main" by using a commit sha or a git tag of the dataset repository.
        use_auth_token (``str`` or ``bool``, optional): Optional string or boolean to use as Bearer token for remote files on the Datasets Hub.
            If True, will get token from `"~/.huggingface"`.
        task (``str``): The task to prepare the dataset for during training and evaluation. Casts the dataset's :class:`Features` to standardized column names and types as detailed in :py:mod:`datasets.tasks`.
        streaming (``bool``, default ``False``): If set to True, don't download the data files. Instead, it streams the data progressively while
            iterating on the dataset. An IterableDataset or IterableDatasetDict is returned instead in this case.

            Note that streaming works for datasets that use data formats that support being iterated over like txt, csv, jsonl for example.
            Json files may be downloaded completely. Also streaming from remote zip or gzip files is supported but other compressed formats
            like rar and xz are not yet supported. The tgz format doesn't allow streaming.
        script_version:
            .. deprecated:: 1.13
                'script_version' was renamed to 'revision' in version 1.13 and will be removed in 1.15.
        **config_kwargs: Keyword arguments to be passed to the :class:`BuilderConfig` and used in the :class:`DatasetBuilder`.
    """

    default_implementation = "default"

    def __init__(  # pylint: disable=too-many-arguments, too-many-locals
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
        revision: Optional[Union[str, Version]] = None,
        use_auth_token: Optional[Union[bool, str]] = None,
        task: Optional[Union[str, TaskTemplate]] = None,
        streaming: bool = False,
        script_version="deprecated",
        **config_kwargs,
    ):
        self._init_params = locals()
        self._init_config_kwargs = config_kwargs

    def read(
        self, split: Optional[Union[str, Split]] = None
    ) -> Union[TrapperDataset, TrapperDatasetDict]:
        """
        Combines the `split` argument with the previously stored parameters
        inside the `__init__` method to load the specified split of the dataset.
        See `datasets.load_dataset`_ for the details of how the dataset is loaded.

        .. _datasets.load_dataset: https://github.com/huggingface/datasets/blob/ec824222c227ea5c9c75568d6f357a819599a6c7/src/datasets/load.py#L1472

        Args:
            split (:class:`Split` or :obj:`str`): Which split of the data to load.
                If None, will return a `dict` with all splits (typically `datasets.Split.TRAIN` and `datasets.Split.TEST`).
                If given, will return a single Dataset.
                Splits can be combined and specified like in tensorflow-datasets.

        Returns:
            :class:`Dataset` or :class:`DatasetDict`:
            - if `split` is not None: the dataset requested,
            - if `split` is None, a ``datasets.DatasetDict`` with each split.

            or :class:`IterableDataset` or :class:`IterableDatasetDict`: if streaming=True

            - if `split` is not None: the dataset requested,
            - if `split` is None, a ``datasets.streaming.IterableDatasetDict`` with each split.
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
