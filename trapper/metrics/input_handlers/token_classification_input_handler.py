import numpy as np
from transformers import EvalPrediction

from trapper.common.constants import IGNORED_LABEL_ID
from trapper.data import LabelMapper
from trapper.metrics.input_handlers import MetricInputHandler


@MetricInputHandler.register("token-classification")
class MetricInputHandlerForTokenClassification(MetricInputHandler):
    """
    `MetricInputHandlerForTokenClassification` provides the conversion of predictions
    and labels from ids to labels by using a `LabelMapper`.

    Args:
        label_mapper (): Required to convert ids to matching labels.
    """

    def __init__(
        self,
        label_mapper: LabelMapper,
    ):
        super(MetricInputHandlerForTokenClassification, self).__init__()
        self._label_mapper = label_mapper

    @property
    def label_mapper(self):
        return self._label_mapper

    def _id_to_label(self, id_: int) -> str:
        return self.label_mapper.get_label(id_)

    def __call__(self, eval_pred: EvalPrediction) -> EvalPrediction:
        predictions, references = eval_pred.predictions, eval_pred.label_ids
        all_predicted_ids = np.argmax(predictions, axis=2)
        all_label_ids = references
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

        processed_eval_pred = EvalPrediction(
            predictions=predictions, label_ids=references
        )
        return processed_eval_pred
