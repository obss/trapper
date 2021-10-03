# pylint: disable=protected-access
"""
This module contains the wrapped task-specific `auto` classes from the
`Transformers` library.
"""
from collections import OrderedDict
from typing import Any, Type

from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForMultipleChoice,
    AutoModelForNextSentencePrediction,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
)

from trapper.models.model import TransformerModel

_COMMON_INPUT_FIELDS = ["input_ids", "attention_mask"]
_TASK_TO_INPUT_FIELDS = OrderedDict(
    [
        # Input fields are required by the data collators and task-specific
        # TransformerModel subclasses.
        (
            "question_answering",
            (
                *_COMMON_INPUT_FIELDS,
                "token_type_ids",
                "start_positions",
                "end_positions",
            ),
        ),
        ("token_classification", (*_COMMON_INPUT_FIELDS, "labels")),
        ("causal_lm", (*_COMMON_INPUT_FIELDS, "token_type_ids", "labels")),
        ("masked_lm", (*_COMMON_INPUT_FIELDS, "token_type_ids", "labels")),
        ("seq2seq_lm", (*_COMMON_INPUT_FIELDS, "labels")),
        (
            "sequence_classification",
            (*_COMMON_INPUT_FIELDS, "token_type_ids", "labels"),
        ),
        ("multiple_choice", (*_COMMON_INPUT_FIELDS, "token_type_ids", "labels")),
        (
            "next_sentence_prediction",
            (*_COMMON_INPUT_FIELDS, "token_type_ids", "labels"),
        ),
    ]
)


def _create_and_register_transformer_subclass(
    auto_cls: Type, task: str
) -> Type[TransformerModel]:
    """
    Dynamically creates a TransformerModel subclass by wrapping an `auto` model
    from the Transformers library. Then, the subclass is registered to
    the framework with the `task` argument and returned.
    Args:
        auto_cls (Type): an `auto` class from `Transformers`
        task (str): registered name of the subclass

    Returns:
        A registered task-specific `TransformerModel` subclass
    """
    cls = _create_transformer_subclass(auto_cls, task)
    TransformerModel.register(task, constructor="from_pretrained")(cls)
    cls._TASK_SPECIFIC_AUTO_CLASS = auto_cls
    cls._TASK_SPECIFIC_FORWARD_PARAMS = _TASK_TO_INPUT_FIELDS[task]
    return cls


def _create_transformer_subclass(
    auto_cls: Type, task: str
) -> Type[TransformerModel]:
    """
    Dynamically creates a TransformerModel subclass by wrapping an `auto` model
    from the Transformers library.
    Args:
        auto_cls (Type): an `auto` class from `Transformers`
        task (str): the task name inserted to the docstring

    Returns:
        A task-specific `TransformerModel` subclass
    """
    auto_cls_name = auto_cls.__name__
    subcls_name = auto_cls_name.replace("Auto", "Transformer")
    attr_dict = {"__doc__": _get_transformer_subclass_doc(auto_cls_name, task)}
    cls: Any = type(subcls_name, (TransformerModel,), attr_dict)
    return cls


def _get_transformer_subclass_doc(auto_model_name: str, task: str):
    return f"""
    Wrapper for `transformers.{auto_model_name}`. Registered as
    the `TransformerModel` factory for `{task}` style tasks.
    """


# Below, we try to add and register as much auto classes as we can from the
# `transformers` library. The original auto classes from `transformers` are found in
# `src/transformers/models/auto/modeling_auto.py` file.
# --------------------------------------------------------------------------

# The classes that have been tested are below

# The base model with a question answering head
TransformerModelForQuestionAnswering = _create_and_register_transformer_subclass(
    AutoModelForQuestionAnswering, "question_answering"
)

# The base model with a token classification head
TransformerModelForTokenClassification = _create_and_register_transformer_subclass(
    AutoModelForTokenClassification, "token_classification"
)
# --------------------------------------------------------------------------

# Experimental classes that have not been tested yet are below. Note that some of
# them may be removed in the future.

# The base model with a causal language modeling head
TransformerModelForCausalLM = _create_and_register_transformer_subclass(
    AutoModelForCausalLM, "causal_lm"
)

# The base model with a masked language modeling head
TransformerModelForMaskedLM = _create_and_register_transformer_subclass(
    AutoModelForMaskedLM, "masked_lm"
)

# The base model with a seq-to-seq language modeling head
TransformerModelForSeq2SeqLM = _create_and_register_transformer_subclass(
    AutoModelForSeq2SeqLM, "seq2seq_lm"
)

# The base model with a sequence classification head
TransformerModelForSequenceClassification = (
    _create_and_register_transformer_subclass(
        AutoModelForSequenceClassification, "sequence_classification"
    )
)

# The base model with a multiple choice head
TransformerModelForMultipleChoice = _create_and_register_transformer_subclass(
    AutoModelForMultipleChoice, "multiple_choice"
)

# The base model with a next sentence prediction head
TransformerModelForNextSentencePrediction = (
    _create_and_register_transformer_subclass(
        AutoModelForNextSentencePrediction, "next_sentence_prediction"
    )
)
