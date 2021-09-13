import json
import logging
from pathlib import Path
from typing import Iterable, Union

from tqdm import tqdm

from trapper.common.constants import SpanTuple
from trapper.common.utils import convert_span_dict_to_tuple
from trapper.data.dataset_readers.dataset_reader import (
    DatasetReader,
    ImproperDataInstanceError,
    IndexedInstance,
)
from trapper.data.dataset_readers.squad.squad_reader import SquadDatasetReader

logger = logging.getLogger(__file__)


@DatasetReader.register("squad-question-answering")
class SquadQuestionAnsweringDatasetReader(SquadDatasetReader):
    NUM_EXTRA_SPECIAL_TOKENS_IN_SEQUENCE = 3  # <bos> context <eos> question <eos>
    MAX_SEQUENCE_LEN = 512

    def text_to_instance(
        self,
        context: str,
        question: SpanTuple,
        paragraph_ind: int = 0,
        answer: SpanTuple = None,
    ) -> IndexedInstance:
        question = self._join_whitespace_prefix(context, question)
        tokenized_context = self._tokenizer.tokenize(context)
        tokenized_question = self._tokenizer.tokenize(question.text)
        self._chop_excess_context_tokens(tokenized_context, tokenized_question)

        instance = {
            "context": self._tokenizer.convert_tokens_to_ids(tokenized_context),
            "question": self._tokenizer.convert_tokens_to_ids(tokenized_question),
        }
        if question.start is not None:
            self._store_tokenized_field_position(
                instance, context, question.start, type_="question"
            )

        if answer is not None:
            answer = self._join_whitespace_prefix(context, answer)
            indexed_question = self._indexed_field(
                context, instance["context"], answer, "answer"
            )
            if indexed_question is None:
                raise ImproperDataInstanceError(
                    "Indexed clue position is out of the bound. Check the input field lengths!"
                )
            instance.update(indexed_question)

        instance["context_index"] = paragraph_ind
        return instance

    def _read(self, file_path: Union[Path, str]) -> Iterable[IndexedInstance]:
        with open(file_path, "r") as in_fp:
            qg_dataset = json.load(in_fp)
        data = qg_dataset["data"]
        logger.info("Number of articles that will be processed: %d.", len(data))
        for article in tqdm(data, desc="articles", total=len(data)):
            paragraphs = article["paragraphs"]
            for paragraph_ind, paragraph in enumerate(paragraphs):
                context = paragraph["context"]
                qa_pairs = paragraph["qas"]
                for qa in qa_pairs:
                    question = {"text": qa["question"], "start": None}
                    question = convert_span_dict_to_tuple(question)
                    if self._is_input_too_long(context, question):
                        continue
                    first_answer = qa.get("answer", None)
                    if first_answer is None:
                        first_answer = qa["answers"][0]
                    # Rename SQuAD answer_start as start for trapper tuple conversion.
                    first_answer["start"] = first_answer["answer_start"]
                    first_answer.pop("answer_start")
                    answer = convert_span_dict_to_tuple(first_answer)
                    try:
                        yield self.text_to_instance(
                            context=context,
                            question=question,
                            paragraph_ind=paragraph_ind,
                            answer=answer,
                        )
                    except ImproperDataInstanceError:
                        continue

    def _is_input_too_long(self, context: str, question: SpanTuple) -> bool:
        context_tokens = self.tokenizer.tokenize(context)
        question_tokens = self.tokenizer.tokenize(question.text)
        return (
            len(context_tokens)
            + len(question_tokens)
            + self.NUM_EXTRA_SPECIAL_TOKENS_IN_SEQUENCE
            > self.MAX_SEQUENCE_LEN
        )
