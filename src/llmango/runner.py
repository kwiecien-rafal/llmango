"""Run orchestration: plan a run from disk, then execute it through a backend."""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from llmango import rows
from llmango.backends import backend_for
from llmango.backends.base import Backend, GenRequest
from llmango.inputs import render
from llmango.manifest import (
    ArmRecord,
    Manifest,
    UsageTotals,
    build_run_id,
    manifest_path,
    write_manifest,
)
from llmango.pricing import PricingEntry, guard_run, load_pricing
from llmango.questions import Arm, PromptTemplate, Question, load_question
from llmango.rows import Generation, Sample
from llmango.spec import ExperimentSpec, schema_name
from llmango.storage import write_results


@dataclass(frozen=True)
class RunPlan:
    """One run, fully built and priced, with nothing sent and no id claimed yet."""

    question: Question
    samples_per_arm: int
    samples: list[Sample]
    price: PricingEntry | None

    @property
    def spec(self) -> ExperimentSpec:
        """The experiment this run's question belongs to."""
        return self.question.spec

    @property
    def samples_total(self) -> int:
        """How many paid calls running this plan would make."""
        return self.samples_per_arm * len(self.question.arms)

    @property
    def requests(self) -> list[GenRequest]:
        """Render each planned sample as the request this run sends for it."""
        return [
            GenRequest(
                model=self.question.model,
                prompt=sample.prompt,
                response_schema=sample.arm.schema,
                temperature=self.question.temperature,
            )
            for sample in self.samples
        ]


@dataclass(frozen=True)
class RunOutcome:
    """What a run wrote, and where it wrote it."""

    manifest: Manifest
    rows_written: int
    parquet_path: Path
    manifest_path: Path

    @property
    def run_id(self) -> str:
        """The id naming this run's files."""
        return self.manifest.run_id


def plan(question_id: str, *, samples_per_arm: int = 1) -> RunPlan:
    """Build one plan for running a question: samples, arms and price."""
    question = load_question(question_id)
    return RunPlan(
        question=question,
        samples_per_arm=samples_per_arm,
        samples=_build_samples(question, samples_per_arm),
        price=_price(question.model),
    )


def run(
    plan: RunPlan, backend: Backend | None = None, *, force: bool = False
) -> RunOutcome:
    """Execute a planned run and persist its results and manifest."""
    question = plan.question
    spec = plan.spec
    price = guard_run(question.model, plan.price, plan.samples_total, force)
    backend = backend or backend_for(question.provider)

    created_at = datetime.now(UTC)
    run_id = build_run_id(question.question_id, created_at)
    if manifest_path(run_id).exists():
        raise ValueError(
            f"Run {run_id} already exists, and a run never overwrites another "
            f"one's files."
        )

    results = backend.generate_many(plan.requests)
    generations = rows.costed(plan.samples, results, price)
    manifest = _manifest(plan, run_id, created_at, price, generations)
    table = rows.build(generations, manifest, spec)
    return RunOutcome(
        manifest=manifest,
        rows_written=len(table),
        parquet_path=write_results(
            table, run_id, question.model, rows.dtypes(spec.extra_raw_dtypes)
        ),
        manifest_path=write_manifest(manifest),
    )


def _manifest(
    plan: RunPlan,
    run_id: str,
    created_at: datetime,
    price: PricingEntry,
    generations: list[Generation],
) -> Manifest:
    """Record what a run was, once it is known what every arm of it used."""
    question = plan.question
    by_arm = rows.usage_by_arm(generations)
    return Manifest(
        run_id=run_id,
        question_id=question.question_id,
        provider=question.provider,
        model=question.model,
        temperature=question.temperature,
        samples_total=plan.samples_total,
        samples_per_arm=plan.samples_per_arm,
        arms=[
            _arm_record(arm, question.templates[arm.lang], by_arm[arm.key])
            for arm in question.arms
        ],
        inputs=question.inputs,
        input_sha256=question.input_sha256,
        pricing=price,
        usage=rows.usage_totals(generations),
        created_at=created_at,
    )


def _arm_record(arm: Arm, template: PromptTemplate, usage: UsageTotals) -> ArmRecord:
    """Record one arm as the manifest pins it: what it asked, and what it used."""
    return ArmRecord(
        lang=arm.lang,
        schema_name=schema_name(arm.schema),
        response_schema=_schema_json(arm.schema),
        template_sha256=template.sha256,
        usage=usage,
    )


def _schema_json(schema: type[BaseModel] | None) -> dict[str, Any] | None:
    """Render a response schema as the JSON stored with the run it was sent in."""
    return schema.model_json_schema() if schema is not None else None


def _build_samples(question: Question, samples_per_arm: int) -> list[Sample]:
    """Render one sample per arm and index from the question's templates."""
    samples: list[Sample] = []
    for arm in question.arms:
        template = question.templates[arm.lang]
        for sample_idx in range(samples_per_arm):
            resolved = question.resolve(arm.lang, sample_idx)
            recorded = {
                name: value.value
                for name, value in resolved.items()
                if value.value is not None
            }
            samples.append(
                Sample(
                    arm=arm,
                    sample_idx=sample_idx,
                    prompt_inputs=json.dumps(recorded, ensure_ascii=False),
                    prompt=render(template.text, resolved),
                )
            )
    return samples


def _price(model: str) -> PricingEntry | None:
    """Look up a model's price, tolerating an absent file so a plan can report it."""
    try:
        table = load_pricing()
    except FileNotFoundError:
        return None
    return table.models.get(model)
