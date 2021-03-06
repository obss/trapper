from typing import Tuple

import numpy as np
import pytest
from transformers import EvalPrediction

from trapper import FIXTURES_ROOT
from trapper.common.io import pickle_load
from trapper.common.utils import is_equal
from trapper.data import TokenizerWrapper
from trapper.metrics import MetricInputHandler

METRIC_FIXTURES = FIXTURES_ROOT / "metrics"


class MockTokenizerWrapper(TokenizerWrapper):
    pass


class MockMetricInputHandler(MetricInputHandler):
    def __init__(self, tokenizer_wrappper: TokenizerWrapper):
        super(MockMetricInputHandler, self).__init__()
        self._tokenizer_wrapper = tokenizer_wrappper

    @property
    def tokenizer(self):
        return self._tokenizer_wrapper.tokenizer

    def __call__(
        self,
        eval_pred: EvalPrediction,
    ) -> EvalPrediction:
        predictions = self.tokenizer.batch_decode(eval_pred.predictions.argmax(-1))
        label_ids = self.tokenizer.batch_decode(eval_pred.label_ids)
        predictions, label_ids = np.array(predictions), np.array(label_ids)
        return EvalPrediction(predictions=predictions, label_ids=label_ids)


@pytest.fixture(scope="function")
def mock_metric_input_handler():
    mock_tokenizer_wrapper = MockTokenizerWrapper.from_pretrained(
        "bert-base-uncased"
    )
    return MockMetricInputHandler(tokenizer_wrappper=mock_tokenizer_wrapper)


@pytest.fixture(scope="function")
def eval_pred():
    predictions_pkl = METRIC_FIXTURES / "predictions.pkl"
    label_ids_pkl = METRIC_FIXTURES / "label_ids.pkl"
    return EvalPrediction(
        predictions=pickle_load(predictions_pkl),
        label_ids=pickle_load(label_ids_pkl),
    )


@pytest.fixture(scope="function")
def actual_predictions():
    return [
        "[unused104] [unused250] [unused207] [unused109] [unused65] [unused49] [unused147] [unused48]",
        "[unused91] [unused241] [unused4] [unused82] [unused237] [unused200] [unused162] [unused227]",
        "[unused184] [unused37] [unused30] [unused219] [unused197] [unused1] [unused53] [unused92]",
    ]


@pytest.fixture(scope="function")
def actual_references():
    return [
        "[unused65] [unused120] [unused212] [unused52] [unused126] [unused149] [unused4] [unused228]",
        "[unused66] [unused22] [unused122] [unused229] [unused181] [unused56] [unused239] [unused73]",
        "[unused75] [unused88] [unused39] [unused182] [unused199] [unused26] [unused152] [unused116]",
    ]


def test_metric_handler(
    eval_pred, mock_metric_input_handler, actual_predictions, actual_references
):
    eval_pred = mock_metric_input_handler(eval_pred)
    predictions = eval_pred.predictions.tolist()
    references = eval_pred.label_ids.tolist()

    assert is_equal(predictions, actual_predictions)
    assert is_equal(references, actual_references)
