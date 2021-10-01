from dataclasses import dataclass
from typing import Dict, Union

import pytest

from trapper.data import DatasetReader, TransformerTokenizer
from trapper.data.dataset_reader import TrapperDataset, TrapperDatasetDict


@dataclass(frozen=True)
class DatasetReaderIdentifier:
    path: str
    name: str


@dataclass(frozen=True)
class RawDatasetIdentifier(DatasetReaderIdentifier):
    split: str


@pytest.fixture(scope="package")  # Used for the tests inside the `data` package
def get_raw_dataset():
    cached_readers: Dict[DatasetReaderIdentifier, DatasetReader] = {}
    cached_datasets: Dict[RawDatasetIdentifier,
                          Union[TrapperDataset, TrapperDatasetDict]] = {}

    def _get_raw_dataset(
            path: str = None, name: str = None, split: str = None
    ) -> Union[TrapperDataset, TrapperDatasetDict]:
        """
        Returns the specified dataset split for testing purposes.

        Args:
            path ():  a local path to processing script or the directory containing
                the script, or a dataset identifier in the HF Datasets Hub
            name (): dataset configuration name, if available.
            split (): one of "train", "validation" or "dev". If `None`, will
                return a dict with all available splits.
        """
        reader_identifier = DatasetReaderIdentifier(path=path, name=name)
        if reader_identifier in cached_readers:
            reader = cached_readers[reader_identifier]
        else:
            reader = DatasetReader(path=path, name=name)
            cached_readers[reader_identifier] = reader

        dataset_identifier = RawDatasetIdentifier(path=path, name=name, split=split)
        if dataset_identifier in cached_datasets:
            dataset = cached_datasets[dataset_identifier]
        else:
            dataset = reader.read(split)
            cached_datasets[dataset_identifier] = dataset

        return dataset

    return _get_raw_dataset


@dataclass
class DataProcessorArguments:
    tokenizer_cls: TransformerTokenizer
    model_type: str = "roberta-base"

    def __post_init__(self):
        if "uncased" in self.model_type:
            self.uncased = True
        else:
            self.uncased = False
        self.tokenizer = self.tokenizer_cls.from_pretrained(self.model_type)
        del self.tokenizer_cls


@pytest.fixture
def get_data_processor_args():
    def _get_data_processor_args(
            tokenizer_cls: TransformerTokenizer = None,
            model_type: str = "roberta-base"
    ) -> DataProcessorArguments:
        return DataProcessorArguments(
            tokenizer_cls=tokenizer_cls,
            model_type=model_type
        )

    return _get_data_processor_args
