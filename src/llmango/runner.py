"""Run orchestration: plan a run from disk, then execute it through a backend."""

import json
import random
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from llmango import rows
from llmango.backends import backend_for
from llmango.backends.base import Backend, GenRequest
from llmango.config import get_manifest_path, get_raw_results_path
from llmango.inputs import render
from llmango.manifest import (
    ArmRecord,
    Manifest,
    UsageTotals,
    build_run_id,
    write_manifest,
)
from llmango.pricing import PricingEntry, guard_run, load_pricing
from llmango.questions import Arm, PromptTemplate, Question, load_question
from llmango.rows import CostedSample, Sample
from llmango.spec import schema_name
from llmango.storage import append_result

ReportProgress = Callable[[Arm, int, int], None]


@dataclass(frozen=True)
class RunPlan:
    """One run, fully built and priced, with nothing sent and no id claimed yet."""

    question: Question
    samples_per_arm: int
    samples: list[Sample]
    price: PricingEntry | None

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
    results_path: Path
    manifest_path: Path

    @property
    def run_id(self) -> str:
        """The id naming this run's files."""
        return self.manifest.run_id

    @property
    def rows_written(self) -> int:
        """How many results reached disk, which a run that died leaves short."""
        return self.manifest.samples_written

    @property
    def finished(self) -> bool:
        """Whether every call the run planned came back and was persisted."""
        return self.manifest.samples_written == self.manifest.samples_total


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
    plan: RunPlan,
    backend: Backend | None = None,
    *,
    force: bool = False,
    report_progress: ReportProgress | None = None,
) -> RunOutcome:
    """Execute a planned run, persisting each result before the next call goes out."""
    question = plan.question
    spec = question.spec
    price = guard_run(question.model, plan.price, plan.samples_total, force)
    backend = backend or backend_for(question.provider)

    created_at = datetime.now(UTC)
    run_id = build_run_id(question.question_id, created_at)
    manifest = _open_manifest(plan, run_id, created_at, price)
    write_manifest(manifest, spec.folder)
    schemas = rows.schema_columns(manifest)

    costed_samples: list[CostedSample] = []
    try:
        for sample, request in zip(plan.samples, plan.requests, strict=True):
            costed_samples.append(
                rows.cost_sample(sample, backend.generate(request), price)
            )
            append_result(
                rows.build_row(costed_samples[-1], manifest, spec, schemas),
                spec.folder,
                run_id,
            )
            manifest = _with_usage(manifest, costed_samples)
            write_manifest(manifest, spec.folder)
            if report_progress is not None:
                report_progress(sample.arm, sample.sample_idx + 1, plan.samples_per_arm)
    except KeyboardInterrupt:
        pass

    return RunOutcome(
        manifest=manifest,
        results_path=get_raw_results_path(spec.folder, run_id),
        manifest_path=get_manifest_path(spec.folder, run_id),
    )


def _open_manifest(
    plan: RunPlan, run_id: str, created_at: datetime, price: PricingEntry
) -> Manifest:
    """Record what a run is about to do, before its first call is paid for."""
    question = plan.question

    return Manifest(
        run_id=run_id,
        question_id=question.question_id,
        provider=question.provider,
        model=question.model,
        temperature=question.temperature,
        samples_total=plan.samples_total,
        samples_per_arm=plan.samples_per_arm,
        arms=[
            _arm_record(arm, question.prompt_templates[arm.lang])
            for arm in question.arms
        ],
        inputs=question.inputs,
        input_sha256=question.input_sha256,
        pricing=price,
        usage=UsageTotals(),
        created_at=created_at,
    )


def _with_usage(manifest: Manifest, costed_samples: list[CostedSample]) -> Manifest:
    """Restate a manifest from everything that has come back so far."""
    by_arm = rows.usage_by_arm(costed_samples)

    return manifest.model_copy(
        update={
            "samples_written": len(costed_samples),
            "arms": [
                arm.model_copy(update={"usage": by_arm.get(arm.key)})
                for arm in manifest.arms
            ],
            "usage": rows.usage_totals(costed_samples),
        }
    )


def _arm_record(arm: Arm, template: PromptTemplate) -> ArmRecord:
    """Record one arm as the manifest pins it, before it has used anything."""
    return ArmRecord(
        lang=arm.lang,
        schema_name=schema_name(arm.schema),
        response_schema=_schema_json(arm.schema),
        template_sha256=template.sha256,
    )


def _schema_json(schema: type[BaseModel] | None) -> dict[str, Any] | None:
    """Render a response schema as the JSON stored with the run it was sent in."""
    return schema.model_json_schema() if schema is not None else None


def _build_samples(question: Question, samples_per_arm: int) -> list[Sample]:
    """Render one sample per arm and index from the question's templates."""
    sample_seeds = [random.getrandbits(64) for _ in range(samples_per_arm)]
    samples: list[Sample] = []
    for arm in question.arms:
        template = question.prompt_templates[arm.lang]
        for sample_idx in range(samples_per_arm):
            resolved = question.resolve(arm.lang, sample_seeds[sample_idx])
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
        pricing_table = load_pricing()
    except FileNotFoundError:
        return None

    return pricing_table.models.get(model)
