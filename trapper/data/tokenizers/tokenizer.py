import os
from copy import deepcopy
from typing import Dict, List, Tuple, Union

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from trapper.common import Params, Registrable
from trapper.common.constants import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN
from trapper.common.utils import add_property


class TokenizerFactory(Registrable):
    """
    The base tokenizer class for trapper that acts as a factory which returns a
    `PreTrainedTokenizerBase` instance after adding some task-specific
    information to it. This includes the task-specific special tokens and the
    maximum sequence length accepted by the model of that tokenizer. This class also
    handles the differences between the special start and end of sequence tokens
    in different models. For example, using this class, you can use
    `tokenizer.bos_token` to access the start token without thinking which model
    you are working on. Otherwise, you would have to use `tokenizer.cls_token` when
    you are working with BERT, or `tokenizer.bos_token` if you are working with
    RoBERTa for example. You may need to override the `TASK_SPECIFIC_SPECIAL_TOKENS`
    class variable to specify the extra special tokens needed for your task.
    Internally, it uses `transformers.AutoTokenizer` for creating the tokenizer
    objects.

    Class variables that can be overridden:

    _TASK_SPECIFIC_SPECIAL_TOKENS (List[str]): A list of extra special tokens that
    is needed for the task at hand. E.g. `CONTEXT` token for SQuAD style question
    answering tasks that utilizes a context.  You can look at
    `QuestionAnsweringTokenizerFactory` for that example.
    """

    default_implementation = "from_pretrained"
    _BOS_TOKEN_KEYS = ("bos_token", "cls_token")
    _EOS_TOKEN_KEYS = ("eos_token", "sep_token")
    _SPECIAL_TOKENS_DICT: Dict[str, str] = {
        "bos_token": BOS_TOKEN,
        "eos_token": EOS_TOKEN,
        "pad_token": PAD_TOKEN,
    }
    _TASK_SPECIFIC_SPECIAL_TOKENS: List[str] = []

    def __init__(self):
        raise EnvironmentError(
            "`TransformerTokenizer` is designed to be instantiated "
            "using the `TransformerTokenizer.from_pretrained` method."
        )

    @classmethod
    def from_params(
        cls,
        params: Params,
        constructor_to_call=None,
        constructor_to_inspect=None,
        **extras,
    ) -> PreTrainedTokenizerBase:
        #  Only used to inform the static type checkers that we return a
        #  `transformers.PreTrainedTokenizerBase`
        return super().from_params(
            params, constructor_to_call, constructor_to_inspect, **extras
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        *inputs,
        **kwargs,
    ) -> PreTrainedTokenizerBase:
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, *inputs, **kwargs
        )
        cls._post_init(tokenizer, **kwargs)
        return tokenizer

    @classmethod
    def _post_init(cls, tokenizer: PreTrainedTokenizerBase, **kwargs):
        tokenizer.__num_added_special_tokens = cls._add_task_specific_tokens(
            tokenizer
        )
        add_property(
            tokenizer,
            {
                "num_added_special_tokens": lambda self: self.__num_added_special_tokens,
                "num_tokens": len,
            },
        )

    @classmethod
    def _add_task_specific_tokens(cls, tokenizer: PreTrainedTokenizerBase) -> int:
        special_tokens_dict = {
            "additional_special_tokens": deepcopy(cls._TASK_SPECIFIC_SPECIAL_TOKENS)
        }
        for tok_name, tok_value in cls._SPECIAL_TOKENS_DICT.items():
            if getattr(tokenizer, tok_name) is None:
                for alternative_pair in (cls._BOS_TOKEN_KEYS, cls._EOS_TOKEN_KEYS):
                    if tok_name in alternative_pair:
                        tok_value = cls._find_alternative_token_value(
                            tok_name, tok_value, alternative_pair
                        )
                        break
                special_tokens_dict[tok_name] = tok_value
        num_added_special_tokens = tokenizer.add_special_tokens(special_tokens_dict)
        return num_added_special_tokens

    @classmethod
    def _find_alternative_token_value(
        cls, token_name: str, token_value: str, alternative_pair: Tuple[str, str]
    ) -> str:
        if token_name == alternative_pair[0]:
            return cls._SPECIAL_TOKENS_DICT.get(alternative_pair[1], token_value)
        else:
            return cls._SPECIAL_TOKENS_DICT.get(alternative_pair[0], token_value)


TokenizerFactory.register("from_pretrained", constructor="from_pretrained")(
    TokenizerFactory
)
