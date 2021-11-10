from __future__ import annotations

from copy import copy
from typing import Dict, Optional, Sequence, Tuple

from trapper.common import Registrable


class LabelMapper(Registrable):
    """
    Used in tasks that require mapping between categorical labels and integer
    ids such as token classification. The list of labels can be provided in the
    constructor as well as the class variable `_LABELS`. The reason for the
    latter is to document the labels and enable writing the labels only once for
    some tasks that have a large number of labels, both of which aim to
    reduce the possibility of errors while specifying the labels.

    Optional class variables:

    _LABELS (Tuple[str]): An optional class variable that may be used for setting
        the list of labels in `LabelMapper.from_labels` method. Note
        that the order you provide will be preserved while assigning ids to the
        labels.
    _IGNORED_LABELS (Tuple[str]): An optional class variable that may be used for
        setting the list of ignored labels.

    Args:
        label_to_id_map (): The mapping from string labels to integer ids
        ignored_labels (): The list of labels to be stored in `ignored_labels`
            attribute. If left as None, the class variable `_IGNORED_LABELS` will
            be used instead.
    """

    default_implementation = "from_labels"
    _LABELS: Tuple[str] = None
    _IGNORED_LABELS: Tuple[str] = ()

    def __init__(
        self,
        label_to_id_map: Optional[Dict[str, int]] = None,
        ignored_labels: Optional[Sequence[str]] = None,
    ):
        #  We need to make `label_to_id_map` optional with default of None,
        #  since otherwise allennlp tries to invoke __init__ although we register
        #  a classmethod as a default constructor and demand it via the "type"
        #  parameter inside the from_params method or a config file.
        if label_to_id_map is None:
            raise ValueError("`label_to_id_map` can not be None!")

        if ignored_labels is None:
            ignored_labels = self._IGNORED_LABELS
        self._ignored_labels = tuple(ignored_labels)

        self._label_to_id_map = label_to_id_map
        self._id_to_label = {
            id_: label for label, id_ in self._label_to_id_map.items()
        }

    @classmethod
    def from_labels(
        cls,
        labels: Optional[Sequence[str]] = None,
        start_id: Optional[int] = 0,
        ignored_labels: Optional[Sequence[str]] = None,
    ) -> LabelMapper:
        """
        Create a `LabelMapper` from a list of labels. The indices will be
        the enumeration of labels starting from `start_ind`.

        Args:
            labels (): The list of labels. Note that the order you provide will
                be preserved while assigning ids to the labels. If left as None,
                the class variable `_LABELS` will be used instead.
            start_id (): The start value for enumeration of label ids. By
                default, we start from 0 and increment.
            ignored_labels (): The list of labels to be stored in `ignored_labels`
                attribute. If left as None, the class variable `_IGNORED_LABELS`
                will be used instead.
        """
        if labels is None:
            labels = cls._LABELS
        ids = tuple(range(start_id, start_id + len(labels)))
        label_to_id = {label: id_ for label, id_ in zip(labels, ids)}
        if ignored_labels is None:
            ignored_labels = cls._IGNORED_LABELS
        return cls(label_to_id, ignored_labels=ignored_labels)

    @property
    def labels(self) -> Tuple[str]:
        return tuple(self._label_to_id_map.keys())

    @property
    def ids(self) -> Tuple[int]:
        return tuple(self._label_to_id_map.values())

    @property
    def label_to_id_map(self) -> Dict[str, int]:
        return copy(self._label_to_id_map)

    @property
    def id_to_label_map(self) -> Dict[int, str]:
        """This method may be used by the pipeline object for inference."""
        return copy(self._id_to_label)

    def get_id(self, label: str) -> int:
        return self._label_to_id_map[label]

    def get_label(self, id_: int) -> str:
        return self._id_to_label[id_]

    @property
    def ignored_labels(self):
        return self._ignored_labels


LabelMapper.register("from_label_to_id_map")(LabelMapper)
LabelMapper.register("from_labels", constructor="from_labels")(LabelMapper)
