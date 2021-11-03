from typing import TypedDict

import numpy as np
import pytest

from trapper.data import TokenizerWrapper
from trapper.metrics import MetricHandler


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
	mock_tokenizer_wrapper = MockTokenizerWrapper.from_pretrained("gpt2")
	return MockMetricHandler(tokenizer_wrapper=mock_tokenizer_wrapper)


@pytest.fixture(scope="function")
def eval_pred():
	n = 3
	m = 8
	h = 50257

	np.random.seed(1)
	predictions = np.random.uniform(size=(n, m, h))
	label_ids = np.random.randint(1, h, size=(n, m))

	return EvalPred(predictions=predictions, label_ids=label_ids)


@pytest.fixture(scope="function")
def actual_predictions():
	return [' Surreyigated complying.) Arthur increment Luther popular', ' Years Chloe horrend ClockbookGazatery imagination', '251 Analytics progressedanticipated outright Auth Projects representation']


@pytest.fixture(scope="function")
def actual_references():
	return [' inappropriateriel covenant subtract Harbor‑ yours dominate', 'essionomed wait inh Rebelladian Sears Messenger', 'Parameter askignore�Albert 1924 upfrontesteem']


def test_metric_handler(eval_pred, mock_metric_handler, actual_predictions, actual_references):
	predictions, references = mock_metric_handler.postprocess(
			predictions=eval_pred.predictions,
			references=eval_pred.label_ids
	)

	assert predictions == actual_predictions
	assert references == actual_references
