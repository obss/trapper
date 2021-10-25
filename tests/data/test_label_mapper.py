import pytest

from trapper.common.utils import is_equal
from trapper.data.label_mapper import LabelMapper

DUMMY_LABELS = ("label1", "label2", "label3")


class MockLabelMapperWithClassVariable(LabelMapper):
    _LABELS = DUMMY_LABELS


class MockLabelMapperWithConstructor(LabelMapper):
    def __init__(self, labels):
        super().__init__(labels)


@pytest.mark.parametrize(
    "start_id",
    [0, 7],
)
def test_mapper_created_from_labels(start_id):
    dummy_ids = (start_id, start_id + 1, start_id + 2)
    dummy_label_to_id_map = {"label1": dummy_ids[0],
                             "label2": dummy_ids[1],
                             "label3": dummy_ids[2]}
    dummy_id_to_label_map = {
        id_: label for label, id_ in dummy_label_to_id_map.items()}

    mapper_from_label_to_id_map = MockLabelMapperWithConstructor(
        dummy_label_to_id_map)
    mapper_from_class_variable_labels = MockLabelMapperWithClassVariable.from_labels(
        start_id=start_id)
    mapper_from_labels = MockLabelMapperWithConstructor.from_labels(
        labels=DUMMY_LABELS, start_id=start_id)

    for mapper in (mapper_from_label_to_id_map,
                   mapper_from_class_variable_labels,
                   mapper_from_labels):
        assert mapper.labels == DUMMY_LABELS
        assert mapper.ids == dummy_ids
        assert is_equal(mapper.label_to_id_map, dummy_label_to_id_map)
        assert is_equal(mapper.id_to_label_map, dummy_id_to_label_map)
        for label, id_ in dummy_label_to_id_map.items():
            assert mapper.get_id(label) == id_
            assert mapper.get_label(id_) == label
