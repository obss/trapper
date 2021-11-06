from typing import Any, Dict, List, Union

import jury
from transformers import EvalPrediction

from trapper.common import Params
from trapper.metrics.metric import Metric, MetricParam
from trapper.metrics.metric_handlers import MetricHandler


@Metric.register("default")
class JuryMetric(Metric):
    def __init__(
        self,
        metric_params: Union[MetricParam, List[MetricParam]],
        metric_handler: MetricHandler,
    ):
        super().__init__(metric_handler=metric_handler)
        self._metric_params = self._convert_metric_params_to_dict(metric_params)

    def __call__(self, pred: EvalPrediction) -> Dict[str, Any]:
        if self._metric_params is None:
            return {}
        jury_scorer = jury.Jury(self._metric_params, run_concurrent=False)

        predictions = pred.predictions
        references = pred.label_ids
        predictions, references = self._metric_handler.preprocess(
            predictions, references
        )

        score = jury_scorer(predictions=predictions, references=references)
        score = self._metric_handler.postprocess(score)

        return score

    def _convert_metric_params_to_dict(
        self, metric_params: Union[MetricParam, List[MetricParam]]
    ) -> Dict:
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
        return converted_metric_params
