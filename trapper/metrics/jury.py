from typing import Any, Dict, List, Optional, Union

import jury
from transformers import EvalPrediction

from trapper.common import Params
from trapper.metrics.metric import Metric, MetricParam


@Metric.register("default")
class JuryMetric(Metric):
    def __init__(
        self,
        metric_params: Union[MetricParam, List[MetricParam]],
        json_normalize: Optional[bool] = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._metric_params = self._convert_metric_params_to_dict(metric_params)
        self.json_normalize = json_normalize

    @property
    def metric_params(self):
        return self._metric_params

    def __call__(self, eval_pred: EvalPrediction) -> Dict[str, Any]:
        if self._metric_params is None:
            return {}
        jury_scorer = jury.Jury(self._metric_params, run_concurrent=False)

        processed_eval_pred = self.input_handler(eval_pred)

        score = jury_scorer(
            predictions=processed_eval_pred.predictions.tolist(),
            references=processed_eval_pred.label_ids.tolist(),
        )
        score = self.output_handler(score)

        if self.json_normalize:
            return self.normalize(score)

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

    @staticmethod
    def normalize(score: Dict) -> Dict:
        extended_results = {}
        for key, value in score.items():
            if isinstance(value, dict):
                for name, val in value.items():
                    extended_results[f"{key}_{name}"] = val
            else:
                extended_results[key] = value
        return extended_results
