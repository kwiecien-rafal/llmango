"""What one experiment declares to the shared pipeline.

An experiment is a grouping, not an identifier. It owns a set of question ids and
everything those questions share: a response schema per variant, a normalization
schema, and a few hooks. Nobody types an experiment's name; every command, path
and function takes a question id, and llmango.experiments maps each id onto the
spec that owns it.

FREE_TEXT_VARIANT and OTHER_CATEGORY are the two engine-wide names every stage
shares. They live here rather than in each stage so that normalize, aggregate and
charts agree on them by construction instead of by three matching string
literals.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass, field

import polars as pl
from pydantic import BaseModel

from llmango.config import sha256_text
from llmango.inputs import BuildInput

FREE_TEXT_VARIANT = "none"
OTHER_CATEGORY = "other"

ExtraRawColumns = Callable[[BaseModel | None, str], dict[str, object]]
ExtraRawDtypes = dict[str, pl.DataType]
ExtraNormalizedColumns = Callable[[pl.DataFrame], dict[str, pl.Series]]


@dataclass(frozen=True)
class SchemaVariant:
    """One way to request an answer: a response schema and its raw-answer field.

    schema is None for the free-text variant, which sends no structured output
    and reads the answer straight from the plain text the model returns.
    """

    schema: type[BaseModel] | None
    field: str | None

    def extract(self, parsed: BaseModel | None, raw_json: str | None) -> str:
        """Return the raw answer string from a parsed model or free text."""
        if self.schema is not None and self.field is not None and parsed is not None:
            return str(getattr(parsed, self.field))
        return raw_json or ""

    @property
    def schema_name(self) -> str | None:
        """The response schema class name, or None for the free-text variant."""
        return self.schema.__name__ if self.schema is not None else None

    @property
    def schema_sha256(self) -> str | None:
        """Hash the variant's JSON schema, or None for the free-text variant.

        The schema is itself part of the prompt: its field names, their order and
        any enums all reach the model. Hashing it makes an edit to a schema as
        traceable as an edit to a prompt template, and keys are left in
        declaration order rather than sorted, because that order is one of the
        things the model sees.
        """
        if self.schema is None:
            return None
        encoded = json.dumps(self.schema.model_json_schema(), ensure_ascii=False)
        return sha256_text(encoded)


@dataclass(frozen=True)
class ExperimentSpec:
    """Everything the generic pipeline needs to run one experiment's questions.

    questions lists the ids the experiment owns, which is what lets one table map
    a question id onto its spec. folder names the prompt tree and the mappings
    directory the experiment keeps its shared files in; it is a location, never a
    reference anyone types.

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
    schema_variants: dict[str, SchemaVariant]
    normalization_schema: type[BaseModel] | None = None
    preprocess: Callable[[str], str] | None = None
    build_input: BuildInput | None = None
    mapping_seed: Callable[[], dict[str, str]] | None = None
    extra_raw_columns: ExtraRawColumns | None = None
    extra_raw_dtypes: ExtraRawDtypes = field(default_factory=ExtraRawDtypes)
    extra_normalized_columns: ExtraNormalizedColumns | None = None

    def variant(self, schema_variant: str) -> SchemaVariant:
        """Return the registered schema variant a run asked for."""
        try:
            return self.schema_variants[schema_variant]
        except KeyError:
            known = ", ".join(sorted(self.schema_variants))
            raise ValueError(
                f"Experiment {self.folder} has no schema variant "
                f"'{schema_variant}'. Known variants: {known}."
            ) from None
