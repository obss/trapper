from allennlp.common import Params as _Params

from trapper.common.utils import append_parent_docstr


@append_parent_docstr
class Params(_Params):
    """
    This class adapts the `Params` class from `allennlp`.
    """
