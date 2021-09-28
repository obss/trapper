"""
This module contains constants like the textual representations of the special
tokens commonly used while training the transformer models on NLP tasks.
Moreover, some NamedTuple and type definitions are supplied for convenience while
working on tasks dealing with spans.
"""
import sys
from typing import NamedTuple

if sys.version_info >= (3, 8):
    from typing import TypedDict  # pylint: disable=no-name-in-module
else:
    from typing_extensions import TypedDict

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

    def to_dict(self):
        return dict(self._asdict())


class SpanDict(TypedDict):
    text: str
    start: int


class PositionTuple(NamedTuple):
    start: int
    end: int

    def to_dict(self):
        return dict(self._asdict())


class PositionDict(TypedDict):
    start: int
    end: int


class Point2D(TypedDict):
    x: int
    y: int
    label: str
