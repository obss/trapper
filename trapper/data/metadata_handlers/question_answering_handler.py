from typing import List, Tuple, Union

import numpy as np

from trapper.data import IndexedInstance
from trapper.data.metadata_handlers import MetadataHandler
from trapper.data.tokenizers import TokenizerWrapper


@MetadataHandler.register("question-answering")
class MetadataHandlerForQuestionAnswering(MetadataHandler):
    _contexts = list()

    def __init__(
        self,
        tokenizer_wrapper: TokenizerWrapper,
    ):
        super().__init__(tokenizer_wrapper)

    def extract_metadata(self, instance: IndexedInstance) -> None:
        context = instance["context"]
        self._contexts.append(context)

    def _decode_answer(self, context: List[int], start, end) -> str:
        answer = context[start - 1 : end - 1]
        return self.tokenizer.decode(answer).lstrip()

    def postprocess(
        self,
        predictions: Union[np.ndarray, Tuple[np.ndarray]],
        references: Union[np.ndarray, Tuple[np.ndarray]],
    ) -> Tuple[List[str], List[str]]:
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
        return predicted_answers, reference_answers
