from abc import ABCMeta, abstractmethod
from typing import Dict, Optional

import numpy as np
from datasets import load_metric
from transformers import EvalPrediction

from trapper.common import Registrable
from trapper.common.constants import IGNORED_LABEL_ID
from trapper.data.label_mapper import LabelMapper


class TransformerMetric(Registrable, metaclass=ABCMeta):
    """
    Base `Registrable` class that is used to register the metrics needed for
    evaluating the models. The subclasses should be implemented as callables
    that accepts a `transformers.EvalPrediction` in their `__call__` method and
    compute score for that prediction.

    Args:
        label_mapper (): Only used in some tasks that require mapping between
            categorical labels and integer ids such as token classification.
    """

    def __init__(self, label_mapper: Optional[LabelMapper] = None):
        self._label_mapper = label_mapper

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
        label_mapper (): Used for mapping between categorical labels and integer
            ids.
        return_entity_level_metrics (bool): Set True to return all the
            entity levels during evaluation. Otherwise, returns overall
            results.
    """

    def __init__(
        self, label_mapper: LabelMapper, return_entity_level_metrics: bool = True
    ):
        if label_mapper is None:
            raise ValueError(
                f"`SeqEvalMetric` can not be instantiated without a `LabelMapper`"
            )

        super().__init__(label_mapper=label_mapper)
        self._metric = load_metric("seqeval")
        self._return_entity_level_metrics = return_entity_level_metrics

    def _id_to_label(self, id_: int) -> str:
        return self._label_mapper.get_label(id_)

    def __call__(self, pred: EvalPrediction) -> Dict[str, float]:
        all_predicted_ids = np.argmax(pred.predictions, axis=2)
        all_label_ids = pred.label_ids
        actual_predictions = []
        actual_labels = []
        for predicted_ids, label_ids in zip(all_predicted_ids, all_label_ids):
            actual_prediction = []
            actual_label = []
            for (p, l) in zip(predicted_ids, label_ids):
                if l != IGNORED_LABEL_ID:
                    actual_prediction.append(self._id_to_label(p))
                    actual_label.append(self._id_to_label(l))

            actual_predictions.append(actual_prediction)
            actual_labels.append(actual_label)

        results: Dict = self._metric.compute(
            predictions=actual_predictions, references=actual_labels
        )
        if self._return_entity_level_metrics:
            return self._extract_entity_level_metrics(results)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    @staticmethod
    def _extract_entity_level_metrics(results: Dict) -> Dict[str, float]:
        extended_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for name, val in value.items():
                    extended_results[f"{key}_{name}"] = val
            else:
                extended_results[key] = value
        return extended_results
