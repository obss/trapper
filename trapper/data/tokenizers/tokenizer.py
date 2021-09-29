from copy import deepcopy
from typing import Tuple, Union

from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
)

from trapper.common import Registrable
from trapper.common.utils import add_property

_PreTrainedTransformerTokenizer = Union[
    PreTrainedTokenizerFast, PreTrainedTokenizer
]


class TransformerTokenizer(Registrable, PreTrainedTokenizerBase):
    """
    The base tokenizer class for trapper that extends `PreTrainedTokenizerBase`
    with further attributes such as task-specific special tokens and the maximum
    sequence length accepted by the model for that tokenizer. To extend this for
    your custom tasks, just override the `_attr_to_special_token` and
    `_TASK_SPECIFIC_SPECIAL_TOKENS` attributes. You can look at
    `QuestionAnsweringTokenizer` for an example.
    """

    default_implementation = "from_pretrained"
    _BOS_TOKEN_KEYS = ("bos_token", "cls_token")
    _EOS_TOKEN_KEYS = ("eos_token", "sep_token")
    _attr_to_special_token = {}
    _TASK_SPECIFIC_SPECIAL_TOKENS = {}

    def __init__(self):
        raise EnvironmentError(
            "`TransformerTokenizer` is designed to be instantiated "
            "using the `TransformerTokenizer.from_pretrained` method."
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, *inputs, **kwargs
        )
        cls._post_init(tokenizer, **kwargs)
        return tokenizer

    @classmethod
    def _post_init(cls, tokenizer: _PreTrainedTransformerTokenizer, **kwargs):
        tokenizer.__num_added_special_tokens = cls._add_task_specific_tokens(
            tokenizer
        )
        tokenizer.__model_max_sequence_length = cls._find_model_max_seq_length(
            tokenizer, kwargs.get("model_max_sequence_length")
        )
        add_property(
            tokenizer,
            {
                "num_added_special_tokens": lambda self: self.__num_added_special_tokens,
                "model_max_sequence_length": lambda self: self.__model_max_sequence_length,
                "num_tokens": len,
            },
        )

    @classmethod
    def _add_task_specific_tokens(
        cls, tokenizer: _PreTrainedTransformerTokenizer
    ) -> int:
        _attr_to_special_token = deepcopy(cls._attr_to_special_token)
        for tok_name, tok_value in cls._TASK_SPECIFIC_SPECIAL_TOKENS.items():
            if getattr(tokenizer, tok_name) is None:
                for alternative_pair in (cls._BOS_TOKEN_KEYS, cls._EOS_TOKEN_KEYS):
                    if tok_name in alternative_pair:
                        tok_value = cls._find_alternative_token_value(
                            tok_name, tok_value, alternative_pair
                        )
                        break
                _attr_to_special_token[tok_name] = tok_value

        return tokenizer.add_special_tokens(_attr_to_special_token)

    @classmethod
    def _find_alternative_token_value(
        cls, token_name: str, token_value: str, alternative_pair: Tuple[str, str]
    ) -> str:
        if token_name == alternative_pair[0]:
            return cls._TASK_SPECIFIC_SPECIAL_TOKENS.get(
                alternative_pair[1], token_value
            )
        else:
            return cls._TASK_SPECIFIC_SPECIAL_TOKENS.get(
                alternative_pair[0], token_value
            )

    @classmethod
    def _find_model_max_seq_length(
        cls,
        tokenizer: _PreTrainedTransformerTokenizer,
        provided_model_max_sequence_length: int = None,
    ) -> int:
        model_max_length = tokenizer.model_max_length
        if provided_model_max_sequence_length is None:
            return model_max_length
        return min(model_max_length, provided_model_max_sequence_length)


TransformerTokenizer.register("from_pretrained", constructor="from_pretrained")(
    TransformerTokenizer
)
