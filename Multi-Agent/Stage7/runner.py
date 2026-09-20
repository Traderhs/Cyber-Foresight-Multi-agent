from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from Stage7.artifact import freeze_stage7_artifact
from Stage7.audits import (
    extract_all_grounding_items,
    extract_all_mediator_items,
    run_grounding_verifier,
    run_mediator_verifier,
    load_cached_audit_responses,
    select_manual_audit_sample,
    select_mediator_audit_sample,
    write_manual_audit_sample,
    write_mediator_audit_sample,
)
from Stage7.axes import (
    axis_a1,
    axis_b1_single_agent,
    axis_b2_policy,
    axis_b3_compute,
    axis_b4_deployment_boundary,
    axis_b5_seed_robustness,
    axis_c1_jurisdiction,
    axis_c2_contextualized_path,
    axis_c3_scenario_provenance,
)
from Stage7.loaders import load_main_case_bundles
from Stage7.schema import AxisResult, EvaluationStatus
from Stage7.variants import load_variant_records


def _combine_axis(
    experiment_id: str,
    name: str,
    scope: str,
    components: list[tuple[str, AxisResult]],
) -> AxisResult:
    statuses = [result.status for _, result in components]
    if any(status == EvaluationStatus.ERROR for status in statuses):
        status = EvaluationStatus.ERROR
    elif statuses and all(status == EvaluationStatus.EVALUATED for status in statuses):
        status = EvaluationStatus.EVALUATED
    elif statuses and all(status == EvaluationStatus.NOT_EVALUATED for status in statuses):
        status = EvaluationStatus.NOT_EVALUATED
    else:
        status = EvaluationStatus.PARTIAL

    metrics = []
    findings: list[str] = []
    limitations: list[str] = []
    provenance: dict[str, Any] = {"components": []}
    for label, result in components:
        metrics.extend(
            metric.model_copy(update={"name": f"{label}.{metric.name}"})
            for metric in result.metrics
        )
        findings.extend(f"{label}: {finding}" for finding in result.findings)
        limitations.extend(f"{label}: {limitation}" for limitation in result.limitations)
        provenance["components"].append(
            {
                "name": label,
                "status": result.status.value,
                "scope": result.scope,
                "provenance": result.provenance,
            }
        )

    return AxisResult(
        experiment_id=experiment_id,
        name=name,
        status=status,
        scope=scope,
        metrics=metrics,
        findings=findings,
        limitations=limitations,
        provenance=provenance,
    )


async def _run_semantic_audits(root: Path, grounding_items, mediator_items):
    # Audit families run sequentially so they never oversubscribe the fixed
    # two-slot llama.cpp runtime.
    grounding = await run_grounding_verifier(root, grounding_items)
    mediator = await run_mediator_verifier(root, mediator_items)
    return grounding, mediator


def run_stage7_validation(
    root: Path,
    *,
    with_llm_audits: bool = False,
) -> tuple[dict[str, Any], Path]:
    root = root.resolve()
    bundles = load_main_case_bundles(root)
    variants = load_variant_records(root)

    grounding_items = extract_all_grounding_items(bundles)
    mediator_items = extract_all_mediator_items(bundles)
    manual_sample_path = write_manual_audit_sample(root, grounding_items)
    mediator_sample_path = write_mediator_audit_sample(root, mediator_items)
    semantic_grounding_items = select_manual_audit_sample(grounding_items)
    semantic_mediator_items = select_mediator_audit_sample(mediator_items)

    grounding_responses: list[dict[str, Any]] = []
    mediator_responses: list[dict[str, Any]] = []
    if with_llm_audits:
        grounding_responses, mediator_responses = asyncio.run(
            _run_semantic_audits(
                root,
                semantic_grounding_items,
                semantic_mediator_items,
            )
        )
    else:
        grounding_responses = load_cached_audit_responses(
            root,
            kind="grounding",
            items=semantic_grounding_items,
        )
        mediator_responses = load_cached_audit_responses(
            root,
            kind="mediator",
            items=semantic_mediator_items,
        )

    prompt_robustness, _ = axis_a1(bundles, variants)
    architecture_baseline, _ = axis_b1_single_agent(bundles, variants)
    contextualized_path, _ = axis_c2_contextualized_path(bundles, variants)

    axes = [
        _combine_axis(
            "A",
            "Robustness",
            "Prompt-paraphrase robustness on the fixed representative three-case subset.",
            [("prompt_paraphrase", prompt_robustness)],
        ),
        _combine_axis(
            "B",
            "Decision architecture and cost",
            "Information-isolated contextual Stage 4 lenses versus one joint three-lens Stage 4 agent, with the frozen upstream Stage 1-3 chain and deterministic downstream policy held fixed; decision-policy dependence and compute/deployment boundaries are reported separately.",
            [
                ("single_agent_baseline", architecture_baseline),
                ("seed_robustness", axis_b5_seed_robustness(bundles, variants)),
                ("decision_policy", axis_b2_policy(bundles)),
                ("compute", axis_b3_compute(root, bundles)),
                ("deployment_boundary", axis_b4_deployment_boundary(bundles)),
            ],
        ),
        _combine_axis(
            "C",
            "Context contribution",
            "Observed jurisdiction sensitivity plus a context-free versus contextualized decision-path comparison; because the contextual path also changes the Stage 3/4 evidence and validation contract, this is not interpreted as a single-factor causal context-only ablation.",
            [
                ("jurisdiction", axis_c1_jurisdiction(bundles)),
                ("contextualized_path", contextualized_path),
                ("scenario_provenance", axis_c3_scenario_provenance(bundles)),
            ],
        ),
    ]

    return freeze_stage7_artifact(
        root,
        bundles=bundles,
        axes=axes,
        grounding_audits=grounding_responses,
        mediator_audits=mediator_responses,
        manual_audit_sample_path=manual_sample_path,
        mediator_audit_sample_path=mediator_sample_path,
    )
