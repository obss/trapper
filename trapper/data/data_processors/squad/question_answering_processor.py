import logging
from typing import Any, Dict

from trapper.common.constants import SpanTuple
from trapper.common.utils import convert_spandict_to_spantuple
from trapper.data.data_processors import DataProcessor
from trapper.data.data_processors.data_processor import (
    ImproperDataInstanceError,
    IndexedInstance,
)
from trapper.data.data_processors.squad.squad_processor import SquadDataProcessor

logger = logging.getLogger(__file__)


@DataProcessor.register("squad-question-answering")
class SquadQuestionAnsweringDataProcessor(SquadDataProcessor):
    NUM_EXTRA_SPECIAL_TOKENS_IN_SEQUENCE = 3  # <bos> context <eos> question <eos>
    MAX_SEQUENCE_LEN = 512

    def process(self, instance_dict: Dict[str, Any]) -> IndexedInstance:
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
    def filtered_instance() -> IndexedInstance:
        return {
            "answer": [-1],
            "answer_position_tokenized": {"start": -1, "end": -1},
            "context": [-1],
            "qa_id": -1,
            "question": [-1],
            "__discard_sample": True,
        }

    def text_to_instance(
        self, context: str, question: SpanTuple, id_: str, answer: SpanTuple = None
    ) -> IndexedInstance:
        question = self._join_whitespace_prefix(context, question)
        tokenized_context = self._tokenizer.tokenize(context)
        tokenized_question = self._tokenizer.tokenize(question.text)
        self._chop_excess_context_tokens(tokenized_context, tokenized_question)

        instance = {
            "context": self._tokenizer.convert_tokens_to_ids(tokenized_context),
            "question": self._tokenizer.convert_tokens_to_ids(tokenized_question),
        }

        if answer is not None:
            answer = self._join_whitespace_prefix(context, answer)
            indexed_answer = self._indexed_field(
                context, instance["context"], field=answer, field_type="answer"
            )
            instance.update(indexed_answer)

        instance["qa_id"] = id_
        return instance

    def _is_input_too_long(self, context: str, question: SpanTuple) -> bool:
        context_tokens = self.tokenizer.tokenize(context)
        question_tokens = self.tokenizer.tokenize(question.text)
        return (
            len(context_tokens)
            + len(question_tokens)
            + self.NUM_EXTRA_SPECIAL_TOKENS_IN_SEQUENCE
            > self.MAX_SEQUENCE_LEN
        )
