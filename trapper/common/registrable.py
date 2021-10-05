"""
`trapper.common.registrable.Registrable` is a "mixin" for allowing registering base
classes and their subclasses so that they can be created by using their
registered name and constructor arguments.
"""

from collections import defaultdict
from typing import ClassVar, DefaultDict

from allennlp.common.registrable import Registrable as _Registrable
from allennlp.common.registrable import _SubclassRegistry

from trapper.common.utils import append_parent_docstr


@append_parent_docstr
class Registrable(_Registrable):
    """
    This class is created to get the registry system from the `allennlp` library
    without the built-in classes registered in `allennlp`. To create a fresh,
    independent registry, we simply extend the `allennlp`'s Registrable class
    and override the class variable `_registry`, which is the actual internal
    registry object.
    """

    _registry: ClassVar[DefaultDict[type, _SubclassRegistry]] = defaultdict(dict)
