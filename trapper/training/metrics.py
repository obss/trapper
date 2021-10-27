from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional, Union

import jury
import numpy as np
from allennlp.common import Params
from datasets import load_metric
from transformers import EvalPrediction

from trapper.common import Registrable
from trapper.common.constants import IGNORED_LABEL_ID
from trapper.data.label_mapper import LabelMapper
from trapper.data.metadata_handlers.metadata_handler import MetadataHandler

MetricParam = Union[str, Dict[str, Any]]


class Metric(Registrable, metaclass=ABCMeta):
    """
    Base `Registrable` class that is used to register the metrics needed for
    evaluating the models. The subclasses should be implemented as callables
    that accepts a `transformers.EvalPrediction` in their `__call__` method and
    compute score for that prediction.

    Args:
        metadata_handler ():
        label_mapper (): Only used in some tasks that require mapping between
            categorical labels and integer ids such as token classification.
    """

    default_implementation = "default"

    def __init__(
        self,
        metadata_handler: MetadataHandler,
    ):
        self._metadata_handler = metadata_handler

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
        super().__init__(
            metadata_handler=metadata_handler
        )
        self._metric_params = metric_params

    def __call__(self, pred: EvalPrediction) -> Dict[str, Any]:
        if self._metric_params is None:
            return {}
        jury_scorer = jury.Jury(self._metric_params, run_concurrent=False)

        predictions = pred.predictions
        references = pred.label_ids
        predictions, references = self._metadata_handler.postprocess(
            predictions, references
        )

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
            metadata_handler=metadata_handler,
        )
