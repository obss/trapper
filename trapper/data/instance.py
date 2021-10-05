from copy import deepcopy
from typing import Mapping, MutableMapping, Sequence, \
    Iterator, Tuple

from overrides import overrides


class Field:
    __slots__ = []  # type: ignore

    def __eq__(self, other) -> bool:
        if isinstance(self, other.__class__):
            # With the way "slots" classes work, self.__slots__ only gives the slots defined
            # by the current class, but not any of its base classes. Therefore to truly
            # check for equality we have to check through all of the slots in all of the
            # base classes as well.
            for class_ in self.__class__.mro():
                for attr in getattr(class_, "__slots__", []):
                    if getattr(self, attr) != getattr(other, attr):
                        return False
            # It's possible that a subclass was not defined as a slots class, in which
            # case we'll need to check __dict__.
            if hasattr(self, "__dict__"):
                return self.__dict__ == other.__dict__
            return True
        return NotImplemented

    def empty_field(self) -> "Field":
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def duplicate(self):
        return deepcopy(self)


class SequenceField(Field):
    """
    A `SequenceField` represents a sequence of things.  This class just adds a
    method onto `Field`: :func:`sequence_length`.     A `ListField` is a list of
    other fields.  You would use this to represent, e.g., a list of
    answer options that are themselves `TextFields`..

    # Parameters

    field_list : `Sequence[Field]`
        A list of `Field` objects to be concatenated into a single input tensor.
            All of the contained `Field` objects must be of the same type.
    """
    __slots__ = ["field_list"]

    def __init__(self, field_list: Sequence[Field]) -> None:
        field_class_set = {field.__class__ for field in field_list}
        assert (
                len(field_class_set) == 1
        ), "ListFields must contain a single field type, found " + str(
            field_class_set)
        # Not sure why mypy has a hard time with this type...
        self.field_list = field_list

    # Sequence[Field] methods
    def __iter__(self) -> Iterator[Field]:
        return iter(self.field_list)

    def __getitem__(self, idx: int) -> Field:
        return self.field_list[idx]

    def __len__(self) -> int:
        return len(self.field_list)

    def sequence_length(self) -> int:
        return len(self.field_list)

    @overrides
    def empty_field(self) -> "SequenceField":
        return SequenceField([self.field_list[0].empty_field()])

    def __str__(self) -> str:
        field_class = self.field_list[0].__class__.__name__
        base_string = f"ListField of {len(self.field_list)} {field_class}s : \n"
        return " ".join(
            [base_string] + [f"\t {field} \n" for field in self.field_list])

    @classmethod
    def from_field(cls, field: Field) -> "SequenceField":
        return cls([field])


class ScalarField(Field):
    __slots__ = ["scalar_value"]

    def __init__(self, scalar_value: int = None) -> None:
        self.scalar_value = scalar_value

    def __len__(self):
        return 1

    def empty_field(self) -> "Field":
        return ScalarField(-1)

    @classmethod
    def null_field(cls) -> "Field":
        return cls(-1)

    def __str__(self) -> str:
        return f"ScalarField({self.scalar_value})"


class BooleanField(ScalarField):
    @classmethod
    def null_field(cls) -> "Field":
        return cls(scalar_value=False)

    def __str__(self) -> str:
        return f"BooleanField({bool(self.scalar_value)})"


class SpanField(Field):
    """
    A `SpanField` is a pair of inclusive, zero-indexed (start, end) indices into a
    :class:`~allennlp.data.fields.sequence_field.SequenceField`, used to represent a
     span of text. Because it's a pair of indices into a :class:`SequenceField`,
     we take one of those as input to make the span's dependence explicit and to
     validate that the span is well defined.

    # Parameters

    span_start : `int`, required.
        The index of the start of the span in the :class:`SequenceField`.
    span_end : `int`, required.
        The inclusive index of the end of the span in the :class:`SequenceField`.
    sequence_field : `SequenceField`, required.
        A field containing the sequence that this `SpanField` is a span inside.
    """

    __slots__ = ["span_start", "span_end", "sequence_field"]

    def __init__(self, span_start: int, span_end: int,
                 sequence_field: SequenceField) -> None:
        self.span_start = span_start
        self.span_end = span_end
        self.sequence_field = sequence_field

        if not isinstance(span_start, int) or not isinstance(span_end, int):
            raise TypeError(
                f"SpanFields must be passed integer indices. Found span indices: "
                f"({span_start}, {span_end}) with types "
                f"({type(span_start)} {type(span_end)})"
            )
        if span_start > span_end:
            raise ValueError(
                f"span_start must be less than span_end, "
                f"but found ({span_start}, {span_end})."
            )

        if span_end > self.sequence_field.sequence_length() - 1:
            raise ValueError(
                f"span_end must be <= len(sequence_length) - 1, but found "
                f"{span_end} and {self.sequence_field.sequence_length() - 1} "
                f"respectively."
            )

    @overrides
    def empty_field(self):
        return SpanField(-1, -1, self.sequence_field.empty_field())

    def __str__(self) -> str:
        return f"SpanField with spans: ({self.span_start}, {self.span_end})."

    def __eq__(self, other) -> bool:
        if isinstance(other, tuple) and len(other) == 2:
            return other == (self.span_start, self.span_end)
        return super().__eq__(other)

    def __len__(self):
        return 2


class Instance(Mapping[str, Field]):
    __slots__ = ["fields"]

    def __init__(self, fields: MutableMapping[str, Field]) -> None:
        self.fields = fields

    def __getitem__(self, key: str) -> Field:
        return self.fields[key]

    def __len__(self) -> int:
        return len(self.fields)

    def __iter__(self):
        return iter(self.fields)

    def __str__(self) -> str:
        base_string = "Instance with fields:\n"
        return " ".join(
            [base_string] + [f"\t {name}: {field} \n" for name, field in
                             self.fields.items()]
        )
