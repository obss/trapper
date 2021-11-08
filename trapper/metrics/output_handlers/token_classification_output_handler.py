from typing import Dict

from trapper.metrics.output_handlers.output_handler import MetricOutputHandler


@MetricOutputHandler.register("token-classification")
class MetricOutputHandlerForTokenClassification(MetricOutputHandler):
    def __init__(self, overall_only: bool = False):
        super(MetricOutputHandlerForTokenClassification, self).__init__()
        self.overall_only = overall_only

    def __call__(self, score: Dict) -> Dict:
        if not self.overall_only:
            return score
        return {
            "precision": score["overall_precision"],
            "recall": score["overall_recall"],
            "f1": score["overall_f1"],
            "accuracy": score["overall_accuracy"],
        }
