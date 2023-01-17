# Copyright 2021 Open Business Software Solutions, the HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
 Question answering pipeline that can be used to extract answer inside a context
 string (e.g. sentence or paragraph) corresponding to a question.

This implementation is adapted from the question answering pipeline from the
HuggingFace's transformers library. Original code is available at:
`<https://github.com/huggingface/transformers/blob/master/src/transformers/pipelines/question_answering.py>`_.

"""
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import (
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    ModelCard,
    PreTrainedTokenizer,
    QuestionAnsweringPipeline,
    SquadExample,
)
from transformers.feature_extraction_utils import PreTrainedFeatureExtractor
from transformers.modeling_utils import PreTrainedModel
from transformers.pipelines import SUPPORTED_TASKS, QuestionAnsweringArgumentHandler
from transformers.pipelines.base import ArgumentHandler, GenericTensor
from transformers.utils import ModelOutput

from trapper.common.constants import SpanDict, SpanTuple
from trapper.common.utils import convert_spandict_to_spantuple
from trapper.data import IndexedInstance
from trapper.data.data_adapters.question_answering_adapter import (
    DataAdapterForQuestionAnswering,
)
from trapper.data.data_collator import DataCollator
from trapper.data.data_processors.squad import SquadQuestionAnsweringDataProcessor
from trapper.models import ModelWrapper
from trapper.pipelines.pipeline import PipelineMixin


class SquadQuestionAnsweringArgumentHandler(QuestionAnsweringArgumentHandler):
    """
    QuestionAnsweringPipeline requires the user to provide multiple arguments (i.e. question & context) to be mapped to
    internal [`SquadExample`].

    QuestionAnsweringArgumentHandler manages all the possible to create a [`SquadExample`] from the command-line
    supplied arguments.
    """

    def normalize(self, item):
        if isinstance(item, SquadExample):
            return item
        elif isinstance(item, dict):
            for k in ["question", "context", "id"]:
                if k not in item:
                    raise KeyError(
                        "You need to provide a dictionary with keys {question:..., context:...}"
                    )
                elif item[k] is None:
                    raise ValueError(f"`{k}` cannot be None")
                elif isinstance(item[k], str) and len(item[k]) == 0:
                    raise ValueError(f"`{k}` cannot be empty")

            return self.create_sample(**item)
        raise ValueError(
            f"{item} argument needs to be of type (SquadExample, dict)"
        )

    def create_sample(
        self,
        id: Union[str, List[str]],
        question: Union[str, List[str]],
        context: Union[str, List[str]],
    ):
        if isinstance(question, list):
            return [
                {"id_": id_, "question": q, "context": c}
                for id_, q, c in zip(id, question, context)
            ]
        else:
            return {"id_": id, "question": question, "context": context}


@PipelineMixin.register(
    "squad-question-answering", constructor="from_partial_objects"
)
class SquadQuestionAnsweringPipeline(PipelineMixin, QuestionAnsweringPipeline):
    """
    Question Answering pipeline using any :obj:`ModelForQuestionAnswering`.

    This question extraction pipeline can currently be loaded from
    :func:`~transformers.pipeline` using the following
    task identifier: :obj:`"question-answering"`.

    The models that this pipeline can use are models that have been
    fine-tuned on a question answering task.
    """

    default_input_names = "question,context"
    _MIN_NULL_SCORE = 1000000
    _LARGE_NEGATIVE = -10000.0

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        data_processor: SquadQuestionAnsweringDataProcessor,
        data_adapter: DataAdapterForQuestionAnswering,
        data_collator: DataCollator,
        feature_extractor: Optional[PreTrainedFeatureExtractor] = None,
        modelcard: Optional[ModelCard] = None,
        framework: Optional[str] = "pt",
        task: str = "question-answering",
        args_parser: ArgumentHandler = None,
        device: int = -1,
        binary_output: bool = False,
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            data_processor=data_processor,
            data_adapter=data_adapter,
            data_collator=data_collator,
            feature_extractor=feature_extractor,
            modelcard=modelcard,
            framework=framework,
            task=task,
            args_parser=args_parser,
            device=device,
            binary_output=binary_output,
        )
        self._args_parser = SquadQuestionAnsweringArgumentHandler()

    def preprocess(
        self, example: Any, **preprocess_kwargs
    ) -> Dict[str, GenericTensor]:
        """
        Preprocessing utilizing data components. This method can be overridden in child
        classes.

        Args:
            example: A dataset, sample of instances or a single instance to be processed.
            **preprocess_kwargs: Additional keyword arguments for preprocess.

        Returns:
            A dictionary making up the model inputs.
        """
        indexed_instance = self.data_processor.text_to_instance(**example)
        indexed_instance = self.data_adapter(indexed_instance)
        yield {
            "indexed_instance": indexed_instance,
            "example": example,
            "is_last": True,
        }

    def _forward(
        self, input_tensors: Dict[str, GenericTensor], **forward_parameters: Dict
    ):
        indexed_instance = input_tensors["indexed_instance"]
        end, start = self._predict_span_scores(indexed_instance)
        return {"start": start, "end": end, **input_tensors}

    def postprocess(
        self,
        model_outputs,
        handle_impossible_answer=False,
        topk=1,
        max_answer_len=15,
    ):
        answers = []
        for output in model_outputs:
            start = output["start"]
            end = output["end"]
            example = output["example"]
            indexed_instance = output["indexed_instance"]
            single_instance_batch = self.data_collator.build_model_inputs(
                (indexed_instance,), should_eliminate_model_incompatible_keys=False
            )
            self.data_collator.pad(single_instance_batch)
            input_ids = np.asarray(single_instance_batch["input_ids"][0])

            # Ensure padded tokens & question tokens cannot belong to the set of candidate answers.
            undesired_tokens = np.ones_like(input_ids)

            min_null_score = self._MIN_NULL_SCORE
            for (start_, end_) in zip(start, end):
                # Normalize logits and spans to retrieve the answer
                start_ = self.normalize_logits(start_).reshape(1, -1)
                end_ = self.normalize_logits(end_).reshape(1, -1)

                if handle_impossible_answer:
                    min_null_score = min(
                        min_null_score, (start_[0] * end_[0]).item()
                    )

                # Mask BOS and EOS tokens
                start_[0, 0] = end_[0, 0] = 0.0

                starts, ends, scores = self.decode(
                    start_, end_, topk, max_answer_len, undesired_tokens
                )

                for start_token_ind, end_tok_ind, score in zip(
                    starts, ends, scores
                ):
                    answers.append(
                        {
                            "score": score.item(),
                            "answer": self._construct_answer(
                                example["context"],
                                input_ids,
                                start_token_ind,
                                end_tok_ind,
                            ),
                        }
                    )

            if handle_impossible_answer:
                answers.append(
                    {
                        "score": min_null_score,
                        "answer": convert_spandict_to_spantuple(
                            {"start": 0, "text": ""}
                        ),
                    }
                )

        answers = self._get_topk_answers(answers, topk)

        if len(answers) == 1:
            return answers[0]
        return answers

    @staticmethod
    def normalize_logits(start_):
        start_ = np.exp(
            start_ - np.log(np.sum(np.exp(start_), axis=-1, keepdims=True))
        )
        return start_

    @staticmethod
    def _get_topk_answers(answers, topk):
        sorted_answers = sorted(answers, key=lambda x: x["score"], reverse=True)
        return sorted_answers[:topk]

    def _predict_span_scores(
        self, indexed_instance: IndexedInstance
    ) -> Tuple[np.ndarray, np.ndarray]:
        fw_args = self.data_collator((indexed_instance,))
        fw_args = {k: v.to(device=self.device) for (k, v) in fw_args.items()}
        # On Windows, the default int type in numpy is np.int32 so we get some non-long tensors.
        fw_args = {
            k: v.long() if v.dtype == torch.int32 else v
            for (k, v) in fw_args.items()
        }
        output = self.model(**fw_args)
        start = output.start_logits.cpu().numpy()
        end = output.end_logits.cpu().numpy()
        return end, start

    def _construct_answer(
        self,
        context: str,
        input_ids: np.ndarray,
        start_token_ind: int,
        end_token_ind: int,
    ) -> SpanTuple:
        answer_start_ind = self._get_answer_start_ind(context, start_token_ind)
        if answer_start_ind is None:
            answer: SpanDict = {
                "start": -1,
                "text": "",
            }
        else:
            answer_token_ids = input_ids[start_token_ind:end_token_ind]
            decoded_answer = self.tokenizer.decode(
                answer_token_ids, skip_special_tokens=True
            ).strip()
            case_corrected_answer = context[
                answer_start_ind : answer_start_ind + len(decoded_answer)
            ]
            answer: SpanDict = {
                "start": answer_start_ind,
                "text": case_corrected_answer,
            }
        return convert_spandict_to_spantuple(answer)

    def _get_answer_start_ind(self, context, start_token_ind):
        context_tokenized = self.tokenizer(context)["input_ids"]
        if start_token_ind >= len(context_tokenized):
            return None

        answer_prefix_token_ids = context_tokenized[0:start_token_ind]
        answer_prefix = self.tokenizer.decode(
            answer_prefix_token_ids, skip_special_tokens=True
        )
        answer_start_ind = len(answer_prefix)

        if context[answer_start_ind] == " ":
            answer_start_ind += 1
        return answer_start_ind


def postprocess_answer(raw_answer: List[Dict]) -> SpanDict:
    """Return the predicted answer with the highest probability"""
    return raw_answer[0]["answer"].to_dict()


SUPPORTED_TASKS["squad-question-answering"] = {
    "impl": SquadQuestionAnsweringPipeline,
    "pt": (ModelWrapper,),
    "tf": (),
}
