import os
from copy import deepcopy
from typing import Dict, List, Tuple, Union

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from trapper.common import Registrable
from trapper.common.constants import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN


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

    def __init__(self, pretrained_tokenizer: PreTrainedTokenizerBase):
        self._pretrained_tokenizer = pretrained_tokenizer
        self._num_added_special_tokens = self._add_task_specific_tokens()

    @property
    def tokenizer(self):
        return self._pretrained_tokenizer

    @property
    def num_added_special_tokens(self) -> int:
        return self._num_added_special_tokens

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        *inputs,
        **kwargs,
    ) -> "TokenizerFactory":
        pretrained_tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, *inputs, **kwargs
        )
        return cls(pretrained_tokenizer)

    def _add_task_specific_tokens(self) -> int:
        tokenizer = self._pretrained_tokenizer
        special_tokens_dict = {
            "additional_special_tokens": deepcopy(
                self._TASK_SPECIFIC_SPECIAL_TOKENS
            )
        }
        for tok_name, tok_value in self._SPECIAL_TOKENS_DICT.items():
            if getattr(tokenizer, tok_name) is None:
                for alternative_pair in (
                    self._BOS_TOKEN_KEYS,
                    self._EOS_TOKEN_KEYS,
                ):
                    if tok_name in alternative_pair:
                        tok_value = self._find_alternative_token_value(
                            tok_name, tok_value, alternative_pair
                        )
                        break
                special_tokens_dict[tok_name] = tok_value
        num_added_special_tokens = tokenizer.add_special_tokens(special_tokens_dict)
        return num_added_special_tokens

    def _find_alternative_token_value(
        self, token_name: str, token_value: str, alternative_pair: Tuple[str, str]
    ) -> str:
        if token_name == alternative_pair[0]:
            return self._SPECIAL_TOKENS_DICT.get(alternative_pair[1], token_value)
        else:
            return self._SPECIAL_TOKENS_DICT.get(alternative_pair[0], token_value)


TokenizerFactory.register("from_pretrained", constructor="from_pretrained")(
    TokenizerFactory
)
