import logging
from typing import Any, Dict

from trapper.common.constants import SpanTuple
from trapper.common.utils import convert_spandict_to_spantuple
from trapper.data.data_processors import DataProcessor
from trapper.data.data_processors.data_processor import (
    ImproperDataInstanceError,
)
from trapper.data.data_processors.squad.squad_processor import SquadDataProcessor
from trapper.data.instance import Instance, SequenceField, ScalarField, \
    BooleanField, SpanField

logger = logging.getLogger(__file__)


@DataProcessor.register("squad-question-answering")
class SquadQuestionAnsweringDataProcessor(SquadDataProcessor):
    NUM_EXTRA_SPECIAL_TOKENS_IN_SEQUENCE = 3  # <bos> context <eos> question <eos>
    MAX_SEQUENCE_LEN = 512

    def process(self, instance_dict: Dict[str, Any]) -> Instance:
        id_ = instance_dict["id"]
        context = instance_dict["context"]
        question = convert_spandict_to_spantuple(
            {"text": instance_dict["question"], "start": -1}
        )
        if self._is_input_too_long(context, question):
            return self.filtered_instance()
        # Rename SQuAD answer_start as start for trapper tuple conversion.
        answers = instance_dict["answers"]
        first_answer = convert_spandict_to_spantuple(
            {"start": answers["answer_start"][0], "text": answers["text"][0]}
        )
        try:
            return self.text_to_instance(
                context=context,
                question=question,
                id_=id_,
                answer=first_answer,
            )
        except ImproperDataInstanceError:
            return self.filtered_instance()

    @staticmethod
    def filtered_instance() -> Instance:
        null_sequence = SequenceField.from_field(ScalarField.null_field())
        return Instance({
            # "answer": SequenceField([ScalarField(-1)]),
            "answer": null_sequence,
            "answer_position_tokenized": SpanField(-1, -1, null_sequence),
            "context": null_sequence,
            "qa_id": ScalarField.null_field(),
            "question": null_sequence,
            "__discard_sample": BooleanField(True),
        })

    def text_to_instance(
            self, context: str, question: SpanTuple, id_: str,
            answer: SpanTuple = None
    ) -> Instance:
        question = self._join_whitespace_prefix(context, question)
        tokenized_context = self._tokenizer.tokenize(context)
        tokenized_question = self._tokenizer.tokenize(question.text)
        self._chop_excess_context_tokens(tokenized_context, tokenized_question)

        instance_dict = {
            "context": self._tokenizer.convert_tokens_to_ids(tokenized_context),
            "question": self._tokenizer.convert_tokens_to_ids(tokenized_question),
        }

        if answer is not None:
            answer = self._join_whitespace_prefix(context, answer)
            indexed_answer = self._indexed_field(
                context, instance_dict["context"], field=answer, field_type="answer"
            )
            instance_dict.update(indexed_answer)

        instance_dict["qa_id"] = id_
        return Instance(instance_dict)

    def _is_input_too_long(self, context: str, question: SpanTuple) -> bool:
        context_tokens = self.tokenizer.tokenize(context)
        question_tokens = self.tokenizer.tokenize(question.text)
        return (
                len(context_tokens)
                + len(question_tokens)
                + self.NUM_EXTRA_SPECIAL_TOKENS_IN_SEQUENCE
                > self.MAX_SEQUENCE_LEN
        )
