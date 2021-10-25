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
from tqdm import tqdm
from transformers import (
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    ModelCard,
    PreTrainedTokenizer,
    add_end_docstrings,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pipelines import SUPPORTED_TASKS
from transformers.pipelines.base import (
    PIPELINE_INIT_ARGS,
    ArgumentHandler,
    Pipeline,
)

from trapper.common.constants import SpanDict, SpanTuple
from trapper.common.utils import convert_spandict_to_spantuple
from trapper.data import IndexedInstance, SquadQuestionAnsweringDataProcessor
from trapper.data.data_adapters.question_answering_adapter import (
    DataAdapterForQuestionAnswering,
)
from trapper.data.data_collator import DataCollator
from trapper.models import ModelWrapper


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

            question = {"text": item["question"], "start": None}
            item["question"] = self._convert_to_span_tuple(question)
            return item
        raise ValueError("{} argument needs to be of type dict".format(item))

    @staticmethod
    def _convert_to_span_tuple(span: Union[SpanDict, SpanTuple]) -> SpanTuple:
        if isinstance(span, SpanDict):
            span = convert_spandict_to_spantuple(span)
        return span

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


@add_end_docstrings(PIPELINE_INIT_ARGS)
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
        modelcard: Optional[ModelCard] = None,
        framework: Optional[str] = None,
        device: int = -1,
        task: str = "",
        **kwargs
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            modelcard=modelcard,
            framework=framework,
            device=device,
            task=task,
            **kwargs,
        )

        self._args_parser = QuestionAnsweringArgumentHandler()
        self.check_model_type(MODEL_FOR_QUESTION_ANSWERING_MAPPING)
        self._data_processor = data_processor
        self._data_adapter = data_adapter
        self._data_collator = data_collator

    def __call__(self, *args, **kwargs):
        """
        Find the answer(s) corresponding to the questions(s) in given context(s).

        Args:
            args (dict or a list of dicts):
                One or several dicts containing the question and context.
            question (:obj:`str` or :obj:`List[str]`): One or several question(s)
                (must be used in conjunction with the :obj:`context` argument).
            context (:obj:`str` or :obj:`List[str]`):
                One or several context(s) associated with the answer(s) (must be
                 used in conjunction with the :obj:`question` argument).
            topk (:obj:`int`, `optional`, defaults to 1):
                The number of answers to return (will be chosen by order of
                likelihood).
            max_answer_len (:obj:`int`, `optional`, defaults to 15):
                The maximum length of predicted answers (e.g., only answers
                with a shorter length are considered).
            max_seq_len (:obj:`int`, `optional`, defaults to 384):
                The maximum length of the total sentence (context)
                after tokenization. The context will be
                split in several chunks (using :obj:`doc_stride`) if needed.
            max_question_len (:obj:`int`, `optional`, defaults to 64):
                The maximum length of the question after tokenization. It will be
                truncated if needed.
            handle_impossible_answer (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not we accept impossible as an answer.

        Return:
            A list of :obj:`dict`: or a list of :obj:`dict`: Each result
            comes as a dictionary with the following keys:
            **score** (:obj:`float`) -- The probability associated to the
                answer.
            **answer** (:obj:`SpanTuple`) -- The answer corresponding to the
                question.
        """
        self._set_default_kwargs(kwargs)
        self._validate_kwargs(kwargs)

        # Convert inputs to features
        examples = self._args_parser(*args, **kwargs)
        all_answers = []
        for example in tqdm(examples, disable=kwargs["disable_tqdm"]):
            indexed_instance = self._data_processor.text_to_instance(**example)
            indexed_instance = self._data_adapter(indexed_instance)
            # Manage tensor allocation on correct device
            with self.device_placement():
                with torch.no_grad():
                    end, start = self._predict_span_scores(indexed_instance)

            single_instance_batch = self._data_collator.build_model_inputs(
                (indexed_instance,), should_eliminate_model_incompatible_keys=False
            )
            self._data_collator.pad(single_instance_batch)
            input_ids = np.asarray(single_instance_batch["input_ids"][0])

            min_null_score = self._MIN_NULL_SCORE
            answers = []
            for (start_, end_) in zip(start, end):
                # Normalize logits and spans to retrieve the answer
                start_ = self.normalize_logits(start_).reshape(1, -1)
                end_ = self.normalize_logits(end_).reshape(1, -1)

                if kwargs["handle_impossible_answer"]:
                    min_null_score = min(
                        min_null_score, (start_[0] * end_[0]).item()
                    )

                # Mask BOS and EOS tokens
                start_[0, 0] = end_[0, 0] = 0.0

                starts, ends, scores = self._decode(
                    start_, end_, kwargs["topk"], kwargs["max_answer_len"]
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

            if kwargs["handle_impossible_answer"]:
                answers.append(
                    {
                        "score": min_null_score,
                        "answer": convert_spandict_to_spantuple(
                            {"start": 0, "text": ""}
                        ),
                    }
                )

            answers = self._get_topk_answers(answers, kwargs)
            all_answers.append(answers)

        if len(all_answers) == 1:
            return all_answers[0]
        return all_answers

    @staticmethod
    def normalize_logits(start_):
        start_ = np.exp(
            start_ - np.log(np.sum(np.exp(start_), axis=-1, keepdims=True))
        )
        return start_

    @staticmethod
    def _get_topk_answers(answers, kwargs):
        sorted_answers = sorted(answers, key=lambda x: x["score"], reverse=True)
        return sorted_answers[: kwargs["topk"]]

    @staticmethod
    def _validate_kwargs(kwargs):
        if kwargs["topk"] < 1:
            raise ValueError(
                "topk parameter should be >= 1 (got {})".format(kwargs["topk"])
            )
        if kwargs["max_answer_len"] < 1:
            raise ValueError(
                "max_answer_len parameter should be >= 1 (got {})".format(
                    kwargs["max_answer_len"]
                )
            )

    def _predict_span_scores(
        self, indexed_instance: IndexedInstance
    ) -> Tuple[np.ndarray, np.ndarray]:
        fw_args = self._data_collator((indexed_instance,))
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

    @staticmethod
    def _set_default_kwargs(kwargs):
        kwargs.setdefault("padding", "longest")
        kwargs.setdefault("topk", 1)
        kwargs.setdefault("max_answer_len", 15)
        kwargs.setdefault("max_seq_len", 384)
        kwargs.setdefault("max_question_len", 64)
        kwargs.setdefault("handle_impossible_answer", False)
        kwargs.setdefault("disable_tqdm", False)

    def _construct_answer(
        self,
        context: str,
        input_ids: np.ndarray,
        start_token_ind: int,
        end_token_ind: int,
    ) -> SpanTuple:
        answer_start_ind = self._get_answer_start_ind(
            context, input_ids, start_token_ind
        )
        answer_inds = list(range(start_token_ind, end_token_ind))
        answer_token_ids = [input_ids[ind] for ind in answer_inds]
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

    def _get_answer_start_ind(self, context, input_ids, start_token_ind):
        answer_prefix_inds = list(range(0, start_token_ind))
        answer_prefix_token_ids = [input_ids[ind] for ind in answer_prefix_inds]
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
    "pt": ModelWrapper,
}
