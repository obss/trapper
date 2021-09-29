import pytest

from trapper.data import DatasetReader


@pytest.fixture
def dataset_reader():
    return DatasetReader(path="squad_qa_test_fixture")


def test_split_names(dataset_reader):
    assert "train" in dataset_reader.split_names
    assert "validation" in dataset_reader.split_names
    assert "test" not in dataset_reader.split_names


def test_get_dataset(dataset_reader):
    assert len(dataset_reader.get_dataset("train")) == 5
    assert len(dataset_reader.get_dataset("validation")) == 6
    with pytest.raises(ValueError):
        dataset_reader.get_dataset("test")
