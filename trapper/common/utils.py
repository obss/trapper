"""
Various utilities for working on data, docstrings etc while using trapper.
"""

from typing import Callable, Dict, List, Type, Union

from deepdiff import DeepDiff

from trapper.common.constants import SpanDict, SpanTuple


def convert_spandict_to_spantuple(span: SpanDict) -> SpanTuple:
    return SpanTuple(text=span["text"], start=span["start"])


def get_docstr(callable_: Union[Type, Callable]) -> str:
    """Returns the docstring of the argument or empty string if it does not
    have any docstring"""
    cls_doc = getattr(callable_, "__doc__", None)
    return "" if cls_doc is None else cls_doc


def append_parent_docstr(cls: Type = None, parent_id: int = 0):
    """
    A decorator that appends the docstring of the decorated class' first parent
    into the decorated class' docstring.

    Args:
        cls : decorated class
        parent_id : the order of the parent in class definition,
            starting from 0. (default=0)
    """

    def cls_wrapper(_cls: Type) -> Type:
        first_parent = _cls.__bases__[parent_id]
        _cls.__doc__ = get_docstr(_cls) + get_docstr(first_parent)
        return _cls

    if cls is None:
        return cls_wrapper
    return cls_wrapper(cls)


def append_callable_docstr(cls: Type, callable_: Union[Type, Callable]):
    """
    A decorator that appends the docstring of a callable into the decorated class'
    docstring.

    Args:
        cls (): decorated class
        callable_ (): The class or function whose docstring is appended

    Returns:

    """

    def cls_wrapper(_cls: Type) -> Type:
        _cls.__doc__ = get_docstr(_cls) + get_docstr(callable_)
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


def is_equal(x: Union[Dict, List], y: Union[Dict, List]) -> bool:
    """Checks equality of two nested container type e.g. list or dict"""
    return not DeepDiff(x, y)
