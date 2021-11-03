import numpy as np
import pytest

from trapper import FIXTURES_ROOT
from trapper.common.io import pickle_load
from trapper.common.utils import is_equal
from trapper.data import TokenizerWrapper
from trapper.metrics import MetricHandler

METRIC_FIXTURES = FIXTURES_ROOT / "metrics"


class MockTokenizerWrapper(TokenizerWrapper):
    pass


class MockMetricHandler(MetricHandler):
    pass


class EvalPred:
    def __init__(self, predictions: np.ndarray, label_ids: np.ndarray):
        self.predictions = predictions
        self.label_ids = label_ids


@pytest.fixture(scope="function")
def mock_metric_handler():
    mock_tokenizer_wrapper = MockTokenizerWrapper.from_pretrained(
        "bert-base-uncased"
    )
    return MockMetricHandler(tokenizer_wrapper=mock_tokenizer_wrapper)


@pytest.fixture(scope="function")
def eval_pred():
    predictions_pkl = METRIC_FIXTURES / "predictions.pkl"
    label_ids_pkl = METRIC_FIXTURES / "label_ids.pkl"
    return EvalPred(
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
    eval_pred, mock_metric_handler, actual_predictions, actual_references
):
    predictions, references = mock_metric_handler.postprocess(
        predictions=eval_pred.predictions, references=eval_pred.label_ids
    )

    assert is_equal(predictions, actual_predictions)
    assert is_equal(references, actual_references)
