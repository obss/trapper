from __future__ import annotations

import os
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from trapper.common import Registrable
from trapper.common.constants import (
    BOS_TOKEN,
    CLS_TOKEN,
    EOS_TOKEN,
    MASK_TOKEN,
    PAD_TOKEN,
    SEP_TOKEN,
    UNK_TOKEN,
)


class TokenizerWrapper(Registrable):
    """
    The base tokenizer class for trapper that acts as a factory which returns a
    `PreTrainedTokenizerBase` instance after adding the task-specific tokens to it.
    Internally, it uses `transformers.AutoTokenizer` for creating the pretrained
    tokenizer objects. In addition to the tokenizer, the wrapper object also holds
    the maximum sequence length accepted by the model of that tokenizer. This class
    also handles the differences between the special start and end of sequence
    tokens in different models. By utilizing a tokenizer wrapped this class, you can
    use `tokenizer.bos_token` to access the start token without thinking which model
    you are working with. Otherwise, you would have to use `tokenizer.cls_token`
    when you are working with BERT, whereas `tokenizer.bos_token` if you are working
    with GPT2 for example. We fill the missing value from the (cls_token, bos_token)
    and (eos_token, sep_token) token pairs by saving the other's value if the
    pretrained tokenizer does not have only one of them. If neither were present,
    they get recorded with separate values. For instance, sep_token is saved with
    the value of eos_token in the GPT2 tokenizer since it has only eos_token
    normally. This is done to make the BOS-CLS and EOS-SEP tokens interchangeable.
    Finally, pad_token, mask_token and unk_token values are also set if they
    were not already present.

    You may need to override the `_TASK_SPECIFIC_SPECIAL_TOKENS` class variable to
    specify the extra special tokens needed for your task.

    Class variables that can be overridden:

    _TASK_SPECIFIC_SPECIAL_TOKENS (List[str]): A list of extra special tokens that
    is needed for the task at hand. E.g. `CONTEXT` token for SQuAD style question
    answering tasks that utilizes a context.  You can look at
    `QuestionAnsweringTokenizerWrapper` for that example.

    Args:
        pretrained_tokenizer (): The pretrained tokenizer to be wrapped
    """

    default_implementation = "from_pretrained"
    _BOS_TOKEN_KEYS = ("bos_token", "cls_token")
    _EOS_TOKEN_KEYS = ("eos_token", "sep_token")
    _SPECIAL_TOKENS_DICT: Dict[str, str] = {
        "bos_token": BOS_TOKEN,
        "eos_token": EOS_TOKEN,
        "cls_token": CLS_TOKEN,
        "sep_token": SEP_TOKEN,
        "pad_token": PAD_TOKEN,
        "mask_token": MASK_TOKEN,
        "unk_token": UNK_TOKEN,
    }
    _TASK_SPECIFIC_SPECIAL_TOKENS: List[str] = []

    def __init__(
        self, pretrained_tokenizer: Optional[PreTrainedTokenizerBase] = None
    ):
        #  We need to make `pretrained_tokenizer` optional with default of None,
        #  since otherwise allennlp tries to invoke __init__ although we
        #  register a classmethod as a default constructor and demand it via the
        #  "type" parameter inside the from_params method or a config file.
        if pretrained_tokenizer is None:
            raise ValueError("`pretrained_tokenizer` can not be None!")
        self._pretrained_tokenizer = pretrained_tokenizer
        self._num_added_special_tokens = self._add_task_specific_tokens()

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
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
    ) -> TokenizerWrapper:
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
        self,
        original_token_name: str,
        original_token_value: str,
        alternative_pair: Tuple[str, str],
    ) -> str:
        if original_token_name == alternative_pair[0]:
            alternative_token_name = alternative_pair[1]
        else:
            alternative_token_name = alternative_pair[0]

        alternative_token_value = getattr(
            self._pretrained_tokenizer, alternative_token_name
        )
        return alternative_token_value or original_token_value


TokenizerWrapper.register("from_pretrained", constructor="from_pretrained")(
    TokenizerWrapper
)
