from copy import copy
from typing import Dict, Optional, Sequence, Tuple

from trapper.common import Registrable


class LabelMapper(Registrable):
    _LABELS: Tuple[str] = None

    def __init__(self, labels: Optional[Sequence[str]] = None):
        """
        Used in tasks that require mapping between categorical labels and integer
        ids such as token classification. The list of labels can be provided in the
        constructor as well as the class variable `_LABELS`. The reason for the
        latter is to document the labels and enable writing the labels only once for
        some tasks that have a large number of labels, both of which aim to
        reduce the possibility of errors while specifying the labels.

        Optional class variables:

        _LABELS: The list of labels. Note that the order you provide will
                be preserved while assigning ids to the labels.

        Args:
            labels (): The list of labels. Note that the order you provide will
                be preserved while assigning ids to the labels.
        """
        self._labels = tuple(labels or self._LABELS)
        self._label_to_id = {label: i for i, label in enumerate(self._labels)}
        self._id_to_label = {i: label for i, label in enumerate(self._labels)}

    @property
    def label_list(self) -> Tuple[str]:
        return self._labels

    @property
    def label_to_id(self) -> Dict[str, int]:
        return copy(self._label_to_id)

    @property
    def id_to_label(self) -> Dict[int, str]:
        """This method may be used by the pipeline object for inference."""
        return copy(self._id_to_label)

    def get_id(self, label: str) -> int:
        return self._label_to_id[label]

    def get_label(self, id_: int) -> str:
        return self._id_to_label[id_]
