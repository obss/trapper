from dataclasses import dataclass
from typing import Dict, Union, Optional

import pytest
from datasets import DownloadConfig
from torch.utils.data import SequentialSampler
from transformers.trainer_pt_utils import SequentialDistributedSampler

from trapper.data import DataCollator, DatasetReader, TransformerTokenizer
from trapper.data.dataset_reader import TrapperDataset, TrapperDatasetDict
from trapper.models.auto_wrappers import _TASK_TO_INPUT_FIELDS


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
    cached_datasets: Dict[
        RawDatasetIdentifier, Union[TrapperDataset, TrapperDatasetDict]
    ] = {}

    def _get_raw_dataset(
            path: str, name: Optional[str] = None, split: Optional[str] = None
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
            reader = DatasetReader(
                path=path, name=name,
                download_config=DownloadConfig(local_files_only=True))
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
    tokenizer_model_name: str = "roberta-base"

    def __post_init__(self):
        if "uncased" in self.tokenizer_model_name:
            self.is_tokenizer_uncased = True
        else:
            self.is_tokenizer_uncased = False
        self.tokenizer = self.tokenizer_cls.from_pretrained(
            self.tokenizer_model_name
        )
        del self.tokenizer_cls


@pytest.fixture
def get_data_processor_args():
    def _get_data_processor_args(
            tokenizer_cls: TransformerTokenizer,
            tokenizer_model_name: str = "roberta-base",
    ) -> DataProcessorArguments:
        return DataProcessorArguments(
            tokenizer_cls=tokenizer_cls, tokenizer_model_name=tokenizer_model_name
        )

    return _get_data_processor_args


@dataclass
class DataCollatorArguments(DataProcessorArguments):
    is_distributed: bool = False
    train_batch_size: int = 2
    task_type: str = "question_answering"
    validation_batch_size: int = 1

    def __post_init__(self):
        super().__post_init__()
        self.model_forward_params = _TASK_TO_INPUT_FIELDS[self.task_type]


@pytest.fixture(scope="package")
def get_data_collator_args():
    def _get_data_collator_args(
            tokenizer_cls: TransformerTokenizer,
            train_batch_size: int,
            validation_batch_size: int,
            tokenizer_model_name: str = "roberta-base",
            task_type: str = "question_answering",
            is_distributed: bool = False,
    ) -> DataProcessorArguments:
        return DataCollatorArguments(
            tokenizer_cls=tokenizer_cls,
            train_batch_size=train_batch_size,
            validation_batch_size=validation_batch_size,
            tokenizer_model_name=tokenizer_model_name,
            task_type=task_type,
            is_distributed=is_distributed,
        )

    return _get_data_collator_args


@pytest.fixture(scope="package")
def get_data_collator():
    def _get_data_collator(args: DataCollatorArguments):
        return DataCollator(args.tokenizer, args.model_forward_params)

    return _get_data_collator


@pytest.fixture
def get_sequential_sampler():
    def _get_sequential_sampler(is_distributed: bool, dataset: TrapperDataset):
        if is_distributed:
            return SequentialDistributedSampler(dataset)
        return SequentialSampler(dataset)  # type: ignore[arg-type]

    return _get_sequential_sampler
