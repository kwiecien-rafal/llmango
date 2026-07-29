"""What one experiment declares to the pipeline, and the names every stage shares."""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import polars as pl
from pydantic import BaseModel

from llmango.inputs import BuildInput

FREE_TEXT = "none"
OTHER_CATEGORY = "other"

ExtraRawColumns = Callable[[BaseModel | None, str], dict[str, object]]
ExtraRawDtypes = dict[str, pl.DataType]
ExtraNormalizedColumns = Callable[[pl.DataFrame], dict[str, pl.Series]]


def answer_field(schema: type[BaseModel]) -> str:
    """Return the one field an answer schema declares, which holds the answer."""
    fields = list(schema.model_fields)
    if len(fields) != 1:
        raise ValueError(
            f"Answer schema {schema.__name__} declares {len(fields)} fields "
            f"({', '.join(fields) or 'none'}); it must declare exactly one, and "
            f"that field is the answer."
        )
    return fields[0]


def canonical_values(schema: type[BaseModel]) -> frozenset[str]:
    """Read the closed category set off a normalization schema's canonical field."""
    field = schema.model_fields.get("canonical")
    annotation = field.annotation if field is not None else None
    if not (isinstance(annotation, type) and issubclass(annotation, Enum)):
        raise ValueError(
            f"Normalization schema {schema.__name__} must declare a canonical field "
            f"typed as an Enum naming every category it may return."
        )
    return frozenset(str(member.value) for member in annotation)


def schema_name(schema: type[BaseModel] | None) -> str | None:
    """The class name a response schema is recorded under, None for free text."""
    return schema.__name__ if schema is not None else None


@dataclass(frozen=True)
class ExperimentSpec:
    """Everything the generic pipeline needs to run one experiment's questions."""

    folder: str
    questions: tuple[str, ...]
    schemas: tuple[type[BaseModel], ...]
    normalization_schema: type[BaseModel] | None = None
    preprocess: Callable[[str], str] | None = None
    build_input: BuildInput | None = None
    mapping_seed: Callable[[], dict[str, str]] | None = None
    extra_raw_columns: ExtraRawColumns | None = None
    extra_raw_dtypes: ExtraRawDtypes = field(default_factory=ExtraRawDtypes)
    extra_normalized_columns: ExtraNormalizedColumns | None = None

    def __post_init__(self) -> None:
        """Reject on import a schema this experiment declares but no stage can read."""
        for schema in self.schemas:
            answer_field(schema)
        if self.normalization_schema is not None:
            canonical_values(self.normalization_schema)

    def schema_named(self, name: str) -> type[BaseModel]:
        """Return the registered response schema a question.yaml names."""
        for schema in self.schemas:
            if schema.__name__ == name:
                return schema
        known = ", ".join(sorted(schema.__name__ for schema in self.schemas))
        raise ValueError(
            f"Experiment {self.folder} registers no schema named '{name}'. "
            f"Known schemas: {known}."
        )
