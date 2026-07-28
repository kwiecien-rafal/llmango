"""What one experiment declares to the shared pipeline.

An experiment is a grouping, not an identifier. It owns a set of question ids and
everything those questions share: the response schemas its questions may be asked
under, a normalization schema, and a few hooks. Nobody types an experiment's name;
every command, path and function takes a question id, and llmango.experiments maps
each id onto the spec that owns it.

FREE_TEXT and OTHER_CATEGORY are the two engine-wide names every stage shares.
FREE_TEXT is what an arm that sends no schema is reported as, since it has no
schema name to go by. They live here rather than in each stage so that normalize,
aggregate and charts agree on them by construction instead of by three matching
string literals.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import polars as pl
from pydantic import BaseModel

from llmango.inputs import BuildInput

FREE_TEXT = "none"
OTHER_CATEGORY = "other"

ExtraRawColumns = Callable[[BaseModel | None, str], dict[str, object]]
ExtraRawDtypes = dict[str, pl.DataType]
ExtraNormalizedColumns = Callable[[pl.DataFrame], dict[str, pl.Series]]


def answer_field(schema: type[BaseModel]) -> str:
    """Return the one field an answer schema declares, which holds the answer.

    An answer schema asks for a single thing, so the field carrying the answer is
    the only field there is. That leaves a question's config nothing to declare
    but the schema itself, and nothing to keep pointing at the answer with.
    """
    fields = list(schema.model_fields)
    if len(fields) != 1:
        raise ValueError(
            f"Answer schema {schema.__name__} declares {len(fields)} fields "
            f"({', '.join(fields) or 'none'}); it must declare exactly one, and "
            f"that field is the answer."
        )
    return fields[0]


def schema_name(schema: type[BaseModel] | None) -> str | None:
    """The class name a response schema is recorded under, None for free text."""
    return schema.__name__ if schema is not None else None


@dataclass(frozen=True)
class ExperimentSpec:
    """Everything the generic pipeline needs to run one experiment's questions.

    questions lists the ids the experiment owns, which is what lets one table map
    a question id onto its spec. folder names the prompt tree and the mappings
    directory the experiment keeps its shared files in; it is a location, never a
    reference anyone types.

    schemas registers the response schemas its questions may be asked under. A
    question.yaml picks from them by class name, so which schema a question used
    is written down once, on the line next to the language it was asked in.

    The pipeline owns a fixed column vocabulary: answer, canonical, is_valid and
    multiple mean the same thing in every experiment and are never renamed, which
    is what lets one query read across the whole published corpus. An experiment
    appends columns of its own through the two extra_ hooks and no more.

    extra_raw_columns adds columns to the raw parquet from one response, and
    extra_raw_dtypes pins their types so a parquet schema never varies with the
    data a run happens to produce. extra_normalized_columns adds columns derived
    from the normalized frame, which is where an experiment computes anything the
    pipeline has no way to know how to compute.

    build_input turns one of the question's declared prompt inputs into the text
    that fills its placeholder. The engine finds and hashes the input's data file
    but never reads inside it, so how a declaration becomes a prompt, including
    any per-sample randomization, is the experiment's own decision.

    mapping_seed offers normalization a label-to-canonical mapping the experiment
    already has, sparing the LLM layer every answer that was on the prompt. The
    experiment covers each of its questions itself, since a question may override
    an input with its own data file and the seed has to reach every list shown.
    """

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
        """Reject a registered schema that is not shaped like an answer schema.

        Checking at declaration means a malformed schema fails on import, before
        any run is planned, rather than once its answers are being extracted.
        """
        for schema in self.schemas:
            answer_field(schema)

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
