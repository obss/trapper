from typing import List, Tuple

import numpy as np
from transformers import EvalPrediction

from trapper.data import IndexedInstance
from trapper.data.tokenizers import TokenizerWrapper
from trapper.metrics.input_handlers import MetricInputHandler


@MetricInputHandler.register("question-answering")
class MetricInputHandlerForQuestionAnswering(MetricInputHandler):
    """
    `MetricInputHandlerForQuestionAnswering` provides the conversion of predictions
    and labels which are the beginning and the end indices to actual answers
    extracted from the context. Since this conversion also requires context, this
    class also overrides `_extract_metadata()` to store context information from
    dataset instances.

    Args:
        tokenizer_wrapper (): Required to convert token ids to strings.
    """

    _contexts = list()

    def __init__(
        self,
        tokenizer_wrapper: TokenizerWrapper,
    ):
        super(MetricInputHandlerForQuestionAnswering, self).__init__()
        self._tokenizer_wrapper = tokenizer_wrapper

    @property
    def tokenizer(self):
        return self._tokenizer_wrapper.tokenizer

    def _extract_metadata(self, instance: IndexedInstance) -> None:
        context = instance["context"]
        self._contexts.append(context)

    def _decode_answer(self, context: List[int], start, end) -> str:
        answer = context[start - 1 : end - 1]
        return self.tokenizer.decode(answer).lstrip()

    def __call__(self, eval_pred: EvalPrediction) -> Tuple[List[str], List[str]]:
        predictions, references = eval_pred.predictions, eval_pred.label_ids
        predicted_starts, predicted_ends = predictions[0].argmax(-1), predictions[
            1
        ].argmax(-1)
        references_starts, references_ends = references[0], references[1]
        n_samples = predictions[0].shape[0]

        predicted_answers = []
        reference_answers = []

        for i in range(n_samples):
            context = self._contexts[i]
            predicted_answer = self._decode_answer(
                context, predicted_starts[i], predicted_ends[i]
            )
            reference_answer = self._decode_answer(
                context, references_starts[i], references_ends[i]
            )
            predicted_answers.append(predicted_answer)
            reference_answers.append(reference_answer)

        predictions = np.array(predicted_answers)
        references = np.array(reference_answers)
        processed_eval_pred = EvalPrediction(
            predictions=predictions, label_ids=references
        )
        return processed_eval_pred
