from dataclasses import dataclass
from typing import Dict, Optional, Type, Union

import pytest
from datasets import DownloadConfig
from torch.utils.data import SequentialSampler
from transformers.trainer_pt_utils import SequentialDistributedSampler

from trapper.data import DataCollator, DatasetReader, TokenizerWrapper
from trapper.data.dataset_reader import TrapperDataset, TrapperDatasetDict
from trapper.models.auto_wrappers import _TASK_TO_INPUT_FIELDS


@dataclass(frozen=True)
class DatasetReaderIdentifier:
    path: str
    name: str


@dataclass(frozen=True)
class RawDatasetIdentifier(DatasetReaderIdentifier):
    split: str


@pytest.fixture(scope="session")
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
                path=path,
                name=name,
                download_config=DownloadConfig(local_files_only=True),
            )
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
    model_max_sequence_length: int = None
    tokenizer_factory: Type[TokenizerWrapper] = TokenizerWrapper
    tokenizer_model_name: str = "roberta-base"
    tokenizer_kwargs: Dict = None

    def __post_init__(self):
        if "uncased" in self.tokenizer_model_name:
            self.is_tokenizer_uncased = True
        else:
            self.is_tokenizer_uncased = False
        tokenizer_kwargs = self.tokenizer_kwargs or {}
        self.tokenizer_wrapper = self.tokenizer_factory.from_pretrained(
            self.tokenizer_model_name, **tokenizer_kwargs
        )
        del self.tokenizer_factory, self.tokenizer_kwargs


@pytest.fixture(scope="session")
def create_data_processor_args():
    def _create_data_processor_args(
        model_max_sequence_length: int = None,
        tokenizer_factory: Type[TokenizerWrapper] = TokenizerWrapper,
        tokenizer_model_name: str = "roberta-base",
        **tokenizer_kwargs,
    ) -> DataProcessorArguments:
        return DataProcessorArguments(
            model_max_sequence_length=model_max_sequence_length,
            tokenizer_factory=tokenizer_factory,
            tokenizer_model_name=tokenizer_model_name,
            tokenizer_kwargs=tokenizer_kwargs,
        )

    return _create_data_processor_args


@dataclass
class DataCollatorArguments(DataProcessorArguments):
    is_distributed: bool = False
    train_batch_size: int = 2
    task_type: str = "question_answering"
    validation_batch_size: int = 1

    def __post_init__(self):
        super().__post_init__()
        self.model_forward_params = _TASK_TO_INPUT_FIELDS[self.task_type]


@pytest.fixture(scope="session")
def create_data_collator_args():
    def _create_data_collator_args(
        train_batch_size: int,
        validation_batch_size: int,
        model_max_sequence_length: int = None,
        tokenizer_factory: Type[TokenizerWrapper] = TokenizerWrapper,
        tokenizer_model_name: str = "roberta-base",
        task_type: str = "question_answering",
        is_distributed: bool = False,
        **tokenizer_kwargs,
    ) -> DataProcessorArguments:
        return DataCollatorArguments(
            model_max_sequence_length=model_max_sequence_length,
            tokenizer_factory=tokenizer_factory,
            tokenizer_kwargs=tokenizer_kwargs,
            train_batch_size=train_batch_size,
            validation_batch_size=validation_batch_size,
            tokenizer_model_name=tokenizer_model_name,
            task_type=task_type,
            is_distributed=is_distributed,
        )

    return _create_data_collator_args


@pytest.fixture(scope="session")
def make_data_collator():
    def _make_data_collator(args: DataCollatorArguments):
        return DataCollator(args.tokenizer_wrapper, args.model_forward_params)

    return _make_data_collator


@pytest.fixture(scope="session")
def make_sequential_sampler():
    def _make_sequential_sampler(is_distributed: bool, dataset: TrapperDataset):
        if is_distributed:
            return SequentialDistributedSampler(dataset)
        return SequentialSampler(dataset)  # type: ignore[arg-type]

    return _make_sequential_sampler
