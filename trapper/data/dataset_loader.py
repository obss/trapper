import logging
from pathlib import Path
from typing import Iterable, Union

import datasets

from trapper.common import Registrable
from trapper.data import DataAdapter
from trapper.data.data_processors.data_processor import DataProcessor, \
    IndexedDataset, IndexedInstance
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

    def load(self, split_name: Union[Path, str]) -> IndexedDataset:
        """
        Reads the dataset for the specified split.

        Args:
            split_name (): one of "train", "validation" or "test"

        Returns:
            an `IndexedDataset` that can be passed to `TransformerTrainer`
        """
        # TODO: Check if `ensure_list` is really necessary?
        raw_data = self.dataset_reader.get_dataset(split_name)
        instances = []
        for instance in self._load(raw_data):
            instances.append(self.data_adapter(instance))
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


DatasetLoader.register("default")(DatasetLoader)
