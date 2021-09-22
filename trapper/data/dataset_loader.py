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
    This class is responsible for reading and pre-processing a dataset.

    Args:
        dataset_reader (): Reads the dataset
        data_processor (): Handles the tokenization and adding the special
            tokens during the pre-processing.
        data_adapter (): Converts the instance into a `IndexedInstance` suitable
            to directly feeding to the model.
    """

    default_implementation = "default"

    def __init__(
            self,
            dataset_reader: DatasetReader,
            data_processor: DataProcessor,
            data_adapter: DataAdapter

    ):
        self._dataset_reader = dataset_reader
        self._data_processor = data_processor
        self._data_adapter = data_adapter

    @property
    def dataset_reader(self):
        return self._dataset_reader

    @property
    def data_processor(self):
        return self._data_processor

    @property
    def data_adapter(self):
        return self._data_adapter

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
        raw_data = self.dataset_reader.get_dataset(split_name)
        processed_data = raw_data.map(self.data_processor)
        return processed_data.map(self.data_adapter)


DatasetLoader.register("default")(DatasetLoader)
