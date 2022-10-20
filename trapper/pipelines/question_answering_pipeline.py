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
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import (
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    ModelCard,
    PreTrainedTokenizer, QuestionAnsweringPipeline,
)
from transformers.feature_extraction_utils import PreTrainedFeatureExtractor
from transformers.modeling_utils import PreTrainedModel
from transformers.pipelines import SUPPORTED_TASKS
from transformers.pipelines.base import GenericTensor
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
from trapper.pipelines import ArgumentHandler
from trapper.pipelines.pipeline import Pipeline


@ArgumentHandler.register("squad-question-answering")
class QuestionAnsweringArgumentHandler(ArgumentHandler):
    """
    QuestionAnsweringPipeline requires the user to provide multiple arguments (
    i.e. answer & context). It supports both single and multiple inputs.
    """

    def normalize(self, item):
        if isinstance(item, dict):
            for k in ["question", "context"]:
                if k not in item:
                    raise KeyError(
                        "You need to provide a dictionary with keys "
                        '("question", "context")'
                    )
                elif item[k] is None:
                    raise ValueError("`{}` cannot be None".format(k))
                elif isinstance(item[k], str) and len(item[k]) == 0:
                    raise ValueError("`{}` cannot be empty".format(k))

            self._add_id(item)
            return item
        raise ValueError("{} argument needs to be of type dict".format(item))

    @staticmethod
    def _convert_to_span_tuple(span: Union[SpanDict, SpanTuple]) -> SpanTuple:
        if isinstance(span, dict):
            span = convert_spandict_to_spantuple(span)
        return span

    @staticmethod
    def _add_id(item: Dict) -> None:
        item["id_"] = item["id"]
        item.pop("id")

    def __call__(self, *args, **kwargs):
        if args is not None and len(args) > 0:
            inputs = self._handle_single_input(args)
        elif "question" in kwargs and "context" in kwargs:
            inputs = self._handle_multiple_inputs(kwargs)
        else:
            raise ValueError("Unknown arguments {}".format(kwargs))

        inputs = self.normalize_inputs(inputs)
        return inputs

    @staticmethod
    def _handle_single_input(args):
        if len(args) == 1:
            inputs = args[0]
        elif len(args) == 2 and {type(el) for el in args} == {str}:
            inputs = [{"question": args[0], "context": args[1]}]
        else:
            inputs = list(args)
        return inputs

    @staticmethod
    def _handle_multiple_inputs(kwargs):
        if isinstance(kwargs["question"], list) and isinstance(
            kwargs["context"], str
        ):
            inputs = [
                {"question": Q, "context": kwargs["context"]}
                for Q in kwargs["question"]
            ]
        elif isinstance(kwargs["question"], list) and isinstance(
            kwargs["context"], list
        ):
            if len(kwargs["question"]) != len(kwargs["context"]):
                raise ValueError(
                    "Questions and contexts don't have the same lengths"
                )

            inputs = [
                {"question": Q, "context": C}
                for Q, C in zip(kwargs["question"], kwargs["context"])
            ]
        else:
            raise ValueError("Arguments can't be understood")
        return inputs

    def normalize_inputs(self, inputs):
        if isinstance(inputs, dict):
            inputs = [inputs]
        elif isinstance(inputs, Iterable):
            # Copy to avoid overriding arguments
            inputs = [i for i in inputs]
        else:
            raise ValueError("Invalid arguments {}".format(inputs))
        for i, item in enumerate(inputs):
            inputs[i] = self.normalize(item)
        return inputs


@Pipeline.register("squad-question-answering", constructor="from_partial_objects")
class SquadQuestionAnsweringPipeline(Pipeline):
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
        framework: Optional[str] = None,
        task: str = "question-answering",
        args_parser: ArgumentHandler = None,
        device: int = -1,
        binary_output: bool = False,
        **kwargs,  # For the ignored arguments
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

        self._args_parser = QuestionAnsweringArgumentHandler()
        self.check_model_type(MODEL_FOR_QUESTION_ANSWERING_MAPPING)

    def _sanitize_parameters(
        self, topk=None, max_clue_len=None, handle_impossible_clue=None, **kwargs
    ):
        postprocess_params = {}
        if topk is not None:
            if topk < 1:
                raise ValueError(
                    "topk parameter should be >= 1 (got {})".format(topk)
                )
            postprocess_params["topk"] = topk
        if max_clue_len is not None:
            if max_clue_len < 1:
                raise ValueError(
                    "max_clue_len parameter should be >= 1 (got {})".format(
                        max_clue_len
                    )
                )
            postprocess_params["max_clue_len"] = max_clue_len
        if handle_impossible_clue is not None:
            postprocess_params["handle_impossible_clue"] = handle_impossible_clue
        return {}, {}, postprocess_params

    def _forward(
        self, input_tensors: Dict[str, GenericTensor], **forward_parameters: Dict
    ) -> ModelOutput:
        indexed_instance = input_tensors["indexed_instance"]
        end, start = self._predict_span_scores(indexed_instance)
        return ModelOutput({"start": start, "end": end, **input_tensors})

    def postprocess(
        self,
        model_outputs,
        handle_impossible_answer=False,
        topk=1,
        max_answer_len=15,
    ):
        start = model_outputs["start"]
        end = model_outputs["end"]
        example = model_outputs["example"]
        indexed_instance = model_outputs["indexed_instance"]
        single_instance_batch = self.data_collator.build_model_inputs(
            (indexed_instance,), should_eliminate_model_incompatible_keys=False
        )
        self.data_collator.pad(single_instance_batch)
        input_ids = np.asarray(single_instance_batch["input_ids"][0])

        min_null_score = self._MIN_NULL_SCORE
        answers = []
        for (start_, end_) in zip(start, end):
            # Normalize logits and spans to retrieve the answer
            start_ = self.normalize_logits(start_).reshape(1, -1)
            end_ = self.normalize_logits(end_).reshape(1, -1)

            if handle_impossible_answer:
                min_null_score = min(min_null_score, (start_[0] * end_[0]).item())

            # Mask BOS and EOS tokens
            start_[0, 0] = end_[0, 0] = 0.0

            starts, ends, scores = self._decode(start_, end_, topk, max_answer_len)

            for start_token_ind, end_tok_ind, score in zip(starts, ends, scores):
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

    @staticmethod
    def _decode(
        start: np.ndarray, end: np.ndarray, topk: int, max_answer_len: int
    ) -> Tuple:
        """
        Take the output of any :obj:`ModelForQuestionAnswering` and will
        generate probabilities for each span to be the actual answer.

        In addition, it filters out some unwanted/impossible cases like answer
        len being greater than max_answer_len. The method supports output the k-best
        answers through the topk argument.

        Args:
            start (:obj:`np.ndarray`): Individual start probabilities
                for each token.
            end (:obj:`np.ndarray`): Individual end probabilities for each token
            topk (:obj:`int`): Indicates how many possible answer span(s) to
                extract from the model output.
            max_answer_len (:obj:`int`): Maximum size of the answer to extract from
                the model's output.
        """
        # Ensure we have batch axis
        if start.ndim == 1:
            start = start[None]

        if end.ndim == 1:
            end = end[None]

        # Compute the score of each tuple(start, end) to be the real answer
        outer = np.matmul(np.expand_dims(start, -1), np.expand_dims(end, 1))

        # Remove candidate with end < start and end - start > max_answer_len
        candidates = np.tril(np.triu(outer), max_answer_len - 1)

        #  Inspired by Chen & al. (https://github.com/facebookresearch/DrQA)
        scores_flat = candidates.flatten()
        if topk == 1:
            idx_sort = [np.argmax(scores_flat)]
        elif len(scores_flat) < topk:
            idx_sort = np.argsort(-scores_flat)
        else:
            idx = np.argpartition(-scores_flat, topk)[0:topk]
            idx_sort = idx[np.argsort(-scores_flat[idx])]

        start, end = np.unravel_index(idx_sort, candidates.shape)[1:]
        return start, end, candidates[0, start, end]


def postprocess_answer(raw_answer: List[Dict]) -> SpanDict:
    """Return the predicted answer with the highest probability"""
    return raw_answer[0]["answer"].to_dict()


SUPPORTED_TASKS["squad-question-answering"] = {
    "impl": SquadQuestionAnsweringPipeline,
    "pt": (ModelWrapper,),
    "tf": (),
}
