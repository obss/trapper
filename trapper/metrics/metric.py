from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Optional, Union

from transformers import EvalPrediction

from trapper.common import Registrable
from trapper.metrics.input_handlers.input_handler import MetricInputHandler
from trapper.metrics.output_handlers import MetricOutputHandler

MetricParam = Union[str, Dict[str, Any]]


class Metric(Registrable, metaclass=ABCMeta):
    """
    Base `Registrable` class that is used to register the metrics needed for
    evaluating the models. The subclasses should be implemented as callables
    that accepts a `transformers.EvalPrediction` in their `__call__` method and
    compute score for that prediction.

    Args:
        input_handler ():
    """

    default_implementation = "default"

    def __init__(
        self,
        input_handler: Optional[MetricInputHandler] = None,
        output_handler: Optional[MetricOutputHandler] = None,
    ):
        self._input_handler = input_handler or MetricInputHandler()
        self._output_handler = output_handler or MetricOutputHandler()

    @property
    def input_handler(self):
        return self._input_handler

    @property
    def output_handler(self):
        return self._output_handler

    @abstractmethod
    def __call__(self, eval_pred: EvalPrediction) -> Dict[str, Any]:
        pass
