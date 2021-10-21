from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Union

import jury
import numpy as np
from allennlp.common import Params
from transformers import EvalPrediction

from trapper.common import Registrable
from trapper.common.constants import PAD_TOKEN_LABEL_ID
from trapper.data import TransformerTokenizer
from trapper.data.metadata_handlers.metadata_handler import MetadataHandler

MetricParam = Union[str, Dict[str, Any]]


class Metric(Registrable, metaclass=ABCMeta):
    """
    Base `Registrable` class that is used to register the metrics needed for
    evaluating the models. The subclasses should be implemented as callables
    that accepts a `transformers.EvalPrediction` in their `__call__` method and
    compute score for that prediction.
    """

    default_implementation = "default"

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Dict[str, Any]:
        pass


@Metric.register("default", constructor="construct_params")
class JuryMetric(Metric):
    def __init__(
        self,
        metric_params: Union[MetricParam, List[MetricParam]],
        metadata_handler: MetadataHandler,
    ):
        self._metric_params = metric_params
        self._metadata_handler = metadata_handler

    def __call__(self, pred: EvalPrediction) -> Dict[str, Any]:
        if self._metric_params is None:
            return {}

        predictions = pred.predictions
        references = pred.label_ids
        predictions, references = self._metadata_handler.postprocess(
                predictions,
                references
        )

        jury_scorer = jury.Jury(self._metric_params, run_concurrent=False)
        return jury_scorer(predictions=predictions, references=references)

    @classmethod
    def construct_params(
        cls,
        metric_params: Union[MetricParam, List[MetricParam]],
        metadata_handler: MetadataHandler,
    ) -> "JuryMetric":
        converted_metric_params = metric_params
        if isinstance(metric_params, Params):
            converted_metric_params = metric_params.params
        elif isinstance(metric_params, list):
            converted_metric_params = []
            for param in metric_params:
                if isinstance(param, Params):
                    metric_param = param.params
                else:
                    metric_param = param
                converted_metric_params.append(metric_param)

        return cls(
            metric_params=converted_metric_params,
            metadata_handler=metadata_handler
        )


@Metric.register("seqeval")
class SeqEvalMetric(Metric):
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
        self,
        label_list: List[str],
        tokenizer: TransformerTokenizer,
        return_entity_level_metrics: bool = True,
    ):
        super().__init__(
            metrics="seqeval", label_list=label_list, tokenizer=tokenizer
        )
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

        results: Dict = self._metrics.compute(
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
