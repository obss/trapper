from abc import ABCMeta, abstractmethod
from typing import Dict, List

import numpy as np
from datasets import load_metric
from transformers import EvalPrediction

from trapper.common import Registrable
from trapper.common.constants import PAD_TOKEN_LABEL_ID


class TransformerMetric(Registrable, metaclass=ABCMeta):
    """
    Base `Registrable` class that is used to register the metrics needed for
    evaluating the models. The subclasses should be implemented as callables
    that accepts a `transformers.EvalPrediction` in their `__call__` method and
    compute score for that prediction.
    """

    @abstractmethod
    def __call__(self, prediction: EvalPrediction) -> Dict[str, float]:
        raise NotImplementedError


@TransformerMetric.register("seqeval")
class SeqEvalMetric(TransformerMetric):
    """
    Creates a token classification task metric that returns of precision,
    recall, f1 and accuracy scores. Internally, uses the "seqeval" metric
    from the HuggingFace's `datasets` library.
    Args:
        return_entity_level_metrics (bool): Set True to return all the
            entity levels during evaluation. Otherwise, returns overall
            results.
    """

    def __init__(
        self, label_list: List[str], return_entity_level_metrics: bool = True
    ):
        self._metric = load_metric("seqeval")
        self._label_list = label_list
        self._return_entity_level_metrics = return_entity_level_metrics

    def __call__(self, pred: EvalPrediction) -> Dict[str, float]:
        all_predicted_ids = np.argmax(pred.predictions, axis=2)
        all_label_ids = pred.label_ids
        actual_predictions = []
        actual_labels = []
        for predicted_ids, label_ids in zip(all_predicted_ids, all_label_ids):
            actual_prediction = []
            actual_label = []
            for (p, l) in zip(predicted_ids, label_ids):
                if l != PAD_TOKEN_LABEL_ID:
                    actual_prediction.append(self._label_list[p])
                    actual_label.append(self._label_list[l])

            actual_predictions.append(actual_prediction)
            actual_labels.append(actual_label)

        results = self._metric.compute(
            predictions=actual_predictions, references=actual_labels
        )
        if self._return_entity_level_metrics:
            return self._extract_entity_level_metrics(results)
        else:
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }

    def _extract_entity_level_metrics(
        self, results: Dict[str, float]
    ) -> Dict[str, float]:
        extended_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for name, val in value.items():
                    extended_results[f"{key}_{name}"] = val
            else:
                extended_results[key] = value
        return extended_results
