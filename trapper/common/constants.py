"""
This module contains constants like the textual representations of the special
tokens commonly used while training the transformer models on NLP tasks.
Moreover, some NamedTuple and type definitions are supplied for convenience while
working on tasks dealing with spans.
"""
from typing import Dict, NamedTuple, Union

PAD_TOKEN_LABEL_ID = -100  # automatically ignored by PyTorch loss functions
CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
PAD_TOKEN = "<pad>"
CONTEXT_TOKEN = "<context>"
ANSWER_TOKEN = "<ans>"


class SpanTuple(NamedTuple):
    text: str
    start: int


class PositionTuple(NamedTuple):
    start: int
    end: int


SpanDict = Dict[str, Union[str, int]]
