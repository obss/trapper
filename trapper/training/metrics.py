from abc import ABCMeta
from typing import Dict, List, Union, Optional

import jury
import numpy as np
from transformers import EvalPrediction

from trapper.common import Registrable
from trapper.common.constants import PAD_TOKEN_LABEL_ID
from trapper.data import TransformerTokenizer


class Metric(Registrable, metaclass=ABCMeta):
    """
    Base `Registrable` class that is used to register the metrics needed for
    evaluating the models. The subclasses should be implemented as callables
    that accepts a `transformers.EvalPrediction` in their `__call__` method and
    compute score for that prediction.
    """

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


@Metric.register("default", constructor="construct_metrics")
class JuryMetric(Metric):

    def __init__(self, label_list: List[jury.metrics.Metric], tokenizer: TransformerTokenizer, metrics: Optional[List[jury.metrics.Metric]] = None):
        self._metrics = metrics
        self._label_list = label_list
        self._tokenizer = tokenizer

    def __call__(self, pred: EvalPrediction) -> Dict[str, float]:
        if self._metrics is None:
            return {}
        all_predicted_ids = np.argmax(pred.predictions, axis=2)
        predictions = [self._tokenizer.decode(pred_id) for pred_id in all_predicted_ids.T]
        references = self._label_list
        jury_scorer = jury.Jury(self._metrics, run_concurrent=True)
        results = jury_scorer(predictions=predictions, references=references)
        return results

    @staticmethod
    def from_list(metrics: List[str]) -> List[jury.metrics.Metric]:
        metrics_ = []
        for metric in metrics:
            metric_ = jury.metrics.load_metric(metric)
            metrics_.append(metric_)

        return metrics_

    @staticmethod
    def from_string(metric: str) -> List[jury.metrics.Metric]:
        metric_ = jury.metrics.load_metric(metric_name=metric)
        return [metric_]

    @classmethod
    def construct_metrics(cls, metrics: Union[str, List[str]], *args, **kwargs):
        if isinstance(metrics, str):
            metrics = cls.from_string(metrics)
        elif isinstance(metrics, list):
            metrics = cls.from_list(metrics)
        return cls(
                metrics=metrics,
                *args,
                **kwargs
        )


# @Metric.register("seqeval")
# class SeqEvalMetric(Metric):
#     """
#     Creates a token classification task metric that returns of precision,
#     recall, f1 and accuracy scores. Internally, uses the "seqeval" metric
#     from the HuggingFace's `datasets` library.
#     Args:
#         return_entity_level_metrics (bool): Set True to return all the
#             entity levels during evaluation. Otherwise, returns overall
#             results.
#     """
#
#     def __init__(
#         self,
#         label_list: List[str],
#         tokenizer: TransformerTokenizer,
#         return_entity_level_metrics: bool = True,
#     ):
#         super().__init__(
#             metrics="seqeval", label_list=label_list, tokenizer=tokenizer
#         )
#         self._return_entity_level_metrics = return_entity_level_metrics
#
#     def __call__(self, pred: EvalPrediction) -> Dict[str, float]:
#         all_predicted_ids = np.argmax(pred.predictions, axis=2)
#         all_label_ids = pred.label_ids
#         actual_predictions = []
#         actual_labels = []
#         for predicted_ids, label_ids in zip(all_predicted_ids, all_label_ids):
#             actual_prediction = []
#             actual_label = []
#             for (p, l) in zip(predicted_ids, label_ids):
#                 if l != PAD_TOKEN_LABEL_ID:
#                     actual_prediction.append(self._label_list[p])
#                     actual_label.append(self._label_list[l])
#
#             actual_predictions.append(actual_prediction)
#             actual_labels.append(actual_label)
#
#         results: Dict = self._metrics.compute(
#             predictions=actual_predictions, references=actual_labels
#         )
#         if self._return_entity_level_metrics:
#             return self._extract_entity_level_metrics(results)
#         return {
#             "precision": results["overall_precision"],
#             "recall": results["overall_recall"],
#             "f1": results["overall_f1"],
#             "accuracy": results["overall_accuracy"],
#         }
#
#     @staticmethod
#     def _extract_entity_level_metrics(results: Dict) -> Dict[str, float]:
#         extended_results = {}
#         for key, value in results.items():
#             if isinstance(value, dict):
#                 for name, val in value.items():
#                     extended_results[f"{key}_{name}"] = val
#             else:
#                 extended_results[key] = value
#         return extended_results
