"""
`Lazy` class is adapted from `allennlp` for use when constructing objects using
`FromParams`, when an argument to a constructor has a sequential dependency with
another argument to the same constructor. See :py:class:`allennlp.common.lazy.Lazy`
for further details.
"""
from allennlp.common.lazy import Lazy as _Lazy

Lazy = _Lazy
