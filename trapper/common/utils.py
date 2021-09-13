"""
Various utilities for working on data, docstrings etc while using trapper.
"""

from typing import Callable, Dict, Type

from trapper.common.constants import SpanDict, SpanTuple


def convert_span_dict_to_tuple(span: SpanDict) -> SpanTuple:
    return SpanTuple(text=span["text"], start=span["start"])


def convert_span_tuple_to_dict(span: SpanTuple) -> SpanDict:
    return {"text": span.text, "start": span.start}


def append_parent_docstr(cls: Type = None, parent_id: int = 0):
    """
    A decorator to append the decorated class' docstring with the
    docstring of the first base class.

    Args:
        cls : decorated class
        parent_id : the order of the parent in class definition,
            starting from 0. (default=0)
    """

    def cls_wrapper(_cls: Type) -> Type:
        first_parent = _cls.__bases__[parent_id]
        cls_doc = getattr(_cls, "__doc__", None)
        cls_doc = "" if cls_doc is None else cls_doc
        _cls.__doc__ = cls_doc + first_parent.__doc__
        return _cls

    if cls is None:
        return cls_wrapper
    return cls_wrapper(cls)


def add_property(inst, name_to_method: Dict[str, Callable]):
    """Dynamically add new properties to an instance by creating a new class
    for the instance that has the additional properties"""
    cls = type(inst)
    # Avoid creating a new class for the inst if it was already done before
    if not hasattr(cls, "__perinstance"):
        cls = type(cls.__name__, (cls,), {})
        cls.__perinstance = True
        inst.__class__ = cls

    for name, method in name_to_method.items():
        setattr(cls, name, property(method))
