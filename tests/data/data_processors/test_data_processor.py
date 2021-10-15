from typing import Any, Dict, List

import datasets
import pytest

from trapper.common.utils import is_equal
from trapper.data import DataProcessor, IndexedInstance


class MockTokenizer:
    @property
    def model_max_length(self):
        return 1

    @staticmethod
    def convert_tokens_to_ids(text: str) -> List[int]:
        return [int(tok.split("token")[-1]) for tok in text]

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return text.split()


class MockTokenizerWrapper:
    def __init__(self):
        self._tokenizer = MockTokenizer()

    @property
    def tokenizer(self):
        return self._tokenizer


class MockDataProcessor(DataProcessor):
    def text_to_instance(self, ind: int, info1: str, info2: str = None
                         ) -> IndexedInstance:
        info1_tokenized = self.tokenizer.tokenize(info1)
        info2_tokenized = self.tokenizer.tokenize(info2)
        return {
            "index": ind,
            "info1": self._tokenizer.convert_tokens_to_ids(info1_tokenized),
            "info2": self._tokenizer.convert_tokens_to_ids(info2_tokenized),
        }

    def process(self, instance_dict: Dict[str, Any]) -> IndexedInstance:
        return self.text_to_instance(
            ind=instance_dict["id"],
            info1=instance_dict["info1"],
            info2=instance_dict["info2_with_suffix"]
        )


@pytest.fixture
def dummy_dataset():
    return datasets.Dataset.from_dict(
        {"id": [0, 1],
         "info1": ["token1 token2", "token3 token4"],
         "info2_with_suffix": ["token4 token5", "token6"]
         }
    )


@pytest.fixture
def mock_processor():
    mock_tokenizer = MockTokenizerWrapper()
    return MockDataProcessor(mock_tokenizer)  # type: ignore


@pytest.mark.parametrize(
    ["index", "expected_instance"],
    [
        (0,
         {"index": 0,
          "info1": [1, 2],
          "info2": [4, 5]
          }
         ),
        (1,
         {"index": 1,
          "info1": [3, 4],
          "info2": [6]
          }
         ),
    ],
)
def test_data_processor(dummy_dataset, mock_processor, index, expected_instance):
    actual_instance = mock_processor(dummy_dataset[index])
    assert is_equal(expected_instance, actual_instance)
