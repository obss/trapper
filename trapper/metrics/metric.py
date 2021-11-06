from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Union

from trapper.common import Registrable
from trapper.metrics.metric_handlers.metric_handler import MetricHandler

MetricParam = Union[str, Dict[str, Any]]


class Metric(Registrable, metaclass=ABCMeta):
    """
    Base `Registrable` class that is used to register the metrics needed for
    evaluating the models. The subclasses should be implemented as callables
    that accepts a `transformers.EvalPrediction` in their `__call__` method and
    compute score for that prediction.

    Args:
        metric_handler ():
    """

    default_implementation = "default"

    def __init__(
        self,
        metric_handler: MetricHandler,
    ):
        self._metric_handler = metric_handler

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Dict[str, Any]:
        pass
