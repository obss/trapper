import pytest
from datasets import DownloadConfig

from trapper.data import DatasetReader


@pytest.fixture
def dataset_reader():
    return DatasetReader(path="squad_qa_test_fixture",
                         download_config=DownloadConfig(local_files_only=True))


def test_read(dataset_reader):
    assert len(dataset_reader.read()) == 2  # dict with two splits
    assert len(dataset_reader.read("train")) == 5
    assert len(dataset_reader.read("validation")) == 6
    assert len(dataset_reader.read("all")) == 11  # splits are combined
    with pytest.raises(ValueError):
        dataset_reader.read("test")
