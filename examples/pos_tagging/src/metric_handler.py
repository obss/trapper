import numpy as np

from trapper.common.constants import IGNORED_LABEL_ID
from trapper.data import LabelMapper, TokenizerWrapper
from trapper.metrics import MetricHandler


@MetricHandler.register("pos-tagging")
class MetricHandlerForPosTagging(MetricHandler):
	def __init__(
			self,
			tokenizer_wrapper: TokenizerWrapper,
			label_mapper: LabelMapper,
	):
		if label_mapper is None:
			raise ValueError(
					f"`SeqEvalMetric` can not be instantiated without a `LabelMapper`"
			)
		super().__init__(
				tokenizer_wrapper=tokenizer_wrapper, label_mapper=label_mapper
		)

	def _id_to_label(self, id_: int) -> str:
		return self.label_mapper.get_label(id_)

	def postprocess(self, predictions, references):
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
		return actual_predictions, actual_labels
