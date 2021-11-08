import logging
from pathlib import Path
from typing import Union

import datasets

from trapper.common import Registrable
from trapper.data import DataAdapter
from trapper.data.data_processors.data_processor import DataProcessor
from trapper.data.dataset_reader import DatasetReader

logger = logging.getLogger(__file__)


class DatasetLoader(Registrable):
    """
    This class is responsible for reading and pre-processing a dataset. This
    involves reading the raw data instances, extracting the task-related fields,
    tokenizing the instances, converting the fields into a format accepted by the
    transformer models as well as taking care of the special tokens. All these
    tasks are performed sequentially by three components in a pipelined manner.

    Args:
        dataset_reader (): Reads the raw dataset.
        data_processor (): Handles pre-processing i.e. tokenization and adding
            the special tokens.
        data_adapter (): Converts the instance into a `IndexedInstance` suitable
            for directly feeding to the models.
    """

    default_implementation = "default"

    def __init__(
        self,
        dataset_reader: DatasetReader,
        data_processor: DataProcessor,
        data_adapter: DataAdapter,
    ):
        self._dataset_reader = dataset_reader
        self._data_processor = data_processor
        self._data_adapter = data_adapter

    @property
    def dataset_reader(self):
        return self._dataset_reader

    @dataset_reader.setter
    def dataset_reader(self, value: DatasetReader):
        if isinstance(value, DatasetReader):
            self._dataset_reader = value
        else:
            raise ValueError(f"The value must be an instance of a {DatasetReader}")

    @property
    def data_processor(self):
        return self._data_processor

    @data_processor.setter
    def data_processor(self, value: DataProcessor):
        if isinstance(value, DataProcessor):
            self._data_processor = value
        else:
            raise ValueError(
                f"The value must be an instance of a "
                f"class derived from {DataProcessor}"
            )

    @property
    def data_adapter(self):
        return self._data_adapter

    @data_adapter.setter
    def data_adapter(self, value: DataAdapter):
        if isinstance(value, DataAdapter):
            self._data_adapter = value
        else:
            raise ValueError(
                f"The value must be an instance of a "
                f"class derived from {DataAdapter}"
            )

    def load(self, split_name: Union[Path, str]) -> datasets.Dataset:
        """
        Reads the specified split of the dataset, process the instances and
        covert them into a format suitable for feeding to a model.

        Args:
            split_name (): one of "train", "validation" or "test"

        Returns:
            a processed split from the dataset, which can be passed to
                `TransformerTrainer`
        """
        raw_data = self.dataset_reader.read(split_name)
        return (
            raw_data.map(self.data_processor)
            .filter(lambda x: not x["__discard_sample"])
            .remove_columns("__discard_sample")
            .map(self.data_adapter)
        )


DatasetLoader.register("default")(DatasetLoader)
