from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from Stage7.config import DECISION_POLICY_VARIANTS
from Stage7.loaders import CaseBundle
from Stage7.logmetrics import aggregate_completion_events, latest_successful_main_log
from Stage7.metrics import (
    cluster_bootstrap_mean_ci,
    mean,
    median,
    pairwise_mean_jaccard,
    population_variance,
)
from Stage7.schema import AxisResult, EvaluationStatus, MetricRecord, VariantResultRecord
from Stage7.variants import baseline_record_map, compare_variant_to_baseline


LENSES = ("technical_feasibility", "institutional_regional", "financial_adoption")


def _metric(
    name: str,
    value: Any,
    *,
    n: int | None = None,
    unit: str | None = None,
    notes: str | None = None,
    ci: tuple[float, float] | None = None,
) -> MetricRecord:
    return MetricRecord(
        name=name,
        value=value,
        n=n,
        unit=unit,
        notes=notes,
        ci_low=ci[0] if ci else None,
        ci_high=ci[1] if ci else None,
    )


def _not_evaluated(
    experiment_id: str,
    name: str,
    scope: str,
    reason: str,
    *,
    provenance: dict[str, Any] | None = None,
) -> AxisResult:
    return AxisResult(
        experiment_id=experiment_id,
        name=name,
        status=EvaluationStatus.NOT_EVALUATED,
        scope=scope,
        limitations=[reason],
        provenance=provenance or {},
    )


def _scenario_rows(bundles: list[CaseBundle]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bundle in bundles:
        for synthesis in bundle.syntheses:
            trace = synthesis.get("lens_evidence_trace") or {}
            rows.append(
                {
                    "case_id": bundle.case_id,
                    "scenario_id": synthesis["scenario_id"],
                    "jurisdiction_code": synthesis["jurisdiction_code"],
                    "recommendation": synthesis["decision_recommendation"],
                    "d_lens": float(synthesis["d_lens"]),
                    "stances": {lens: int(trace[lens]["stance"]) for lens in LENSES},
                    "evidence_ids_by_lens": {
                        lens: list(
                            dict.fromkeys(
                                evidence_id
                                for claim in trace[lens].get("claims") or []
                                for evidence_id in claim.get("evidence_ids") or []
                            )
                        )
                        for lens in LENSES
                    },
                }
            )
    return rows


def _variant_result(
    *,
    experiment_id: str,
    name: str,
    axis_name: str,
    bundles: list[CaseBundle],
    variants: list[VariantResultRecord],
    variant_ids: set[str] | None = None,
) -> tuple[AxisResult, list[dict[str, Any]]]:
    baseline = baseline_record_map(bundles)
    selected = [
        record
        for record in variants
        if record.axis == axis_name
        and (variant_ids is None or record.variant_id in variant_ids)
    ]
    if not selected:
        return (
            _not_evaluated(
                experiment_id,
                name,
                "configured frozen scenario variants",
                f"No {axis_name} variant artifacts are present. No score is fabricated.",
            ),
            [],
        )

    comparisons: list[dict[str, Any]] = []
    missing_baseline: list[str] = []
    for record in selected:
        baseline_scenario_id = str(record.provenance.get("baseline_scenario_id") or record.scenario_id)
        main = baseline.get((record.case_id, baseline_scenario_id))
        if main is None:
            missing_baseline.append(f"{record.case_id}/{baseline_scenario_id}")
            continue
        comparisons.append(compare_variant_to_baseline(record, main))

    stance_rows = [row for row in comparisons if row["stance_agreement"] is not None]
    evidence_rows = [row for row in comparisons if row["evidence_overlap"] is not None]
    direction_rows = [row for row in comparisons if row["claim_direction_overlap"] is not None]
    stance_values = [float(row["stance_agreement"]) for row in stance_rows]
    evidence_values = [float(row["evidence_overlap"]) for row in evidence_rows]
    direction_values = [float(row["claim_direction_overlap"]) for row in direction_rows]
    recommendation_values = [1.0 if row["recommendation_consistent"] else 0.0 for row in comparisons]
    d_values = [float(row["d_lens_abs_delta"]) for row in comparisons]
    stance_clusters = [str(row["case_id"]) for row in stance_rows]
    evidence_clusters = [str(row["case_id"]) for row in evidence_rows]
    direction_clusters = [str(row["case_id"]) for row in direction_rows]
    recommendation_clusters = [str(row["case_id"]) for row in comparisons]
    bootstrap_note = (
        "95% case-cluster bootstrap CI: resample case_id clusters with replacement "
        "(2,000 repetitions; seed=20260916) so KR/EU/US scenarios from one case are not "
        "treated as independent observations."
    )
    status = EvaluationStatus.EVALUATED if comparisons and not missing_baseline else EvaluationStatus.PARTIAL

    return (
        AxisResult(
            experiment_id=experiment_id,
            name=name,
            status=status,
            scope=f"{len(comparisons)} variant-scenario comparisons",
            metrics=[
                _metric(
                    "stance_agreement_mean",
                    mean(stance_values),
                    n=len(stance_values),
                    ci=cluster_bootstrap_mean_ci(stance_values, stance_clusters),
                    notes=bootstrap_note,
                ),
                _metric(
                    "evidence_id_overlap_mean",
                    mean(evidence_values),
                    n=len(evidence_values),
                    ci=cluster_bootstrap_mean_ci(evidence_values, evidence_clusters),
                    notes=bootstrap_note,
                ),
                _metric(
                    "claim_direction_overlap_mean",
                    mean(direction_values),
                    n=len(direction_values),
                    ci=cluster_bootstrap_mean_ci(direction_values, direction_clusters),
                    notes=bootstrap_note,
                ),
                _metric(
                    "recommendation_consistency_rate",
                    mean(recommendation_values),
                    n=len(recommendation_values),
                    ci=cluster_bootstrap_mean_ci(
                        recommendation_values,
                        recommendation_clusters,
                    ),
                    notes=bootstrap_note,
                ),
                _metric("d_lens_abs_delta_median", median(d_values), n=len(d_values)),
            ],
            findings=[
                f"Compared {len(comparisons)} frozen variant outputs to their exact main scenario outputs."
            ],
            limitations=(
                [f"Variant records without exact baseline match were excluded: {missing_baseline}"]
                if missing_baseline
                else []
            ),
            provenance={
                "axis_name": axis_name,
                "variant_ids": sorted({record.variant_id for record in selected}),
                "bootstrap": {
                    "method": "case_cluster_bootstrap",
                    "cluster_key": "case_id",
                    "repetitions": 2000,
                    "seed": 20260916,
                    "alpha": 0.05,
                    "unique_case_count": len({str(row["case_id"]) for row in comparisons}),
                },
            },
        ),
        comparisons,
    )


def axis_a1(
    bundles: list[CaseBundle],
    variants: list[VariantResultRecord],
) -> tuple[AxisResult, list[dict[str, Any]]]:
    return _variant_result(
        experiment_id="A1",
        name="Prompt paraphrase sensitivity",
        axis_name="A1_PROMPT_PARAPHRASE",
        bundles=bundles,
        variants=variants,
    )


def axis_b1_single_agent(
    bundles: list[CaseBundle],
    variants: list[VariantResultRecord],
) -> tuple[AxisResult, list[dict[str, Any]]]:
    return _variant_result(
        experiment_id="B1",
        name="Joint three-lens Stage 4 baseline",
        axis_name="B_STAGE4_ARCHITECTURE_BASELINE",
        bundles=bundles,
        variants=variants,
        variant_ids={"SINGLE_AGENT_BASELINE"},
    )


def _main_policy(t: int, i: int, f: int) -> str:
    if t == -1 or i == -1:
        return "DO_NOT_RECOMMEND"
    if t == 0 or i == 0:
        return "HOLD"
    if f == 1:
        return "RECOMMEND"
    if f == 0:
        return "RECOMMEND_PILOT"
    return "HOLD"


def _strict_policy(t: int, i: int, f: int) -> str:
    if t == -1 or i == -1:
        return "DO_NOT_RECOMMEND"
    if t == 0 or i == 0:
        return "HOLD"
    return "RECOMMEND" if f == 1 else "HOLD"


def _permissive_policy(t: int, i: int, f: int) -> str:
    if t == -1 or i == -1:
        return "DO_NOT_RECOMMEND"
    if t == 0 or i == 0:
        return "HOLD"
    return "RECOMMEND" if f == 1 else "RECOMMEND_PILOT"


def axis_b2_policy(bundles: list[CaseBundle]) -> AxisResult:
    rows = _scenario_rows(bundles)
    details: list[dict[str, Any]] = []
    invariant_count = 0
    main_mismatch: list[str] = []
    for row in rows:
        stances = row["stances"]
        t = stances["technical_feasibility"]
        i = stances["institutional_regional"]
        f = stances["financial_adoption"]
        main = _main_policy(t, i, f)
        strict = _strict_policy(t, i, f)
        permissive = _permissive_policy(t, i, f)
        if main != row["recommendation"]:
            main_mismatch.append(f"{row['case_id']}/{row['scenario_id']}")
        invariant = len({main, strict, permissive}) == 1
        invariant_count += int(invariant)
        details.append(
            {
                "case_id": row["case_id"],
                "scenario_id": row["scenario_id"],
                "stances": stances,
                "main": main,
                "strict": strict,
                "permissive": permissive,
                "policy_invariant": invariant,
            }
        )
    if main_mismatch:
        raise ValueError(f"Current Stage 6 recommendation does not match main policy: {main_mismatch}")
    return AxisResult(
        experiment_id="B2",
        name="Decision-policy sensitivity",
        status=EvaluationStatus.EVALUATED,
        scope=f"{len(rows)} main DecisionObjects x 3 predeclared policies",
        metrics=[
            _metric("policy_invariant_rate", invariant_count / len(rows), n=len(rows)),
            _metric("policy_sensitive_count", len(rows) - invariant_count, n=len(rows)),
            _metric("scenario_policy_results", details, n=len(rows)),
        ],
        findings=[
            "Policy-sensitive recommendations are reported as dependence on the explicit decision rule, not hidden as model failures."
        ],
        provenance={"policy_variants": list(DECISION_POLICY_VARIANTS)},
    )


def axis_b3_compute(root: Path, bundles: list[CaseBundle]) -> AxisResult:
    parsed = latest_successful_main_log(root)
    runtime_profiles = {bundle.case_id: bundle.stage6.get("runtime") for bundle in bundles}
    if parsed is None:
        return _not_evaluated(
            "B3",
            "Compute / deployment burden",
            "main Stage 1-6 runtime",
            "No successful final main-run log exists.",
            provenance={"stage6_runtime_profiles": runtime_profiles},
        )
    aggregate = aggregate_completion_events(parsed["completion_events"])
    return AxisResult(
        experiment_id="B3",
        name="Compute / deployment burden",
        status=EvaluationStatus.PARTIAL,
        scope="latest successful resume run",
        metrics=[
            _metric("llm_completion_event_count", aggregate["calls"], n=aggregate["calls"]),
            _metric("repair_completion_event_count", aggregate["repair_calls"], n=aggregate["calls"]),
            _metric("prompt_tokens", aggregate["prompt_tokens"], unit="tokens"),
            _metric("completion_tokens", aggregate["completion_tokens"], unit="tokens"),
            _metric("total_tokens", aggregate["total_tokens"], unit="tokens"),
            _metric("summed_request_elapsed_seconds", aggregate["elapsed_seconds_sum"], unit="seconds"),
            _metric("by_stage", aggregate["by_stage"]),
        ],
        findings=[
            "The frozen runtime identity records the model, quantization, llama.cpp build, context allocation, parallel slots, seed, decoding parameters, and repair policy."
        ],
        limitations=[
            "The latest successful run reused five Stage 6 case artifacts and therefore is not a clean from-scratch full-pipeline cost benchmark.",
            "Peak VRAM/memory is not reconstructed from the existing log and requires a dedicated clean cost run if needed."
        ],
        provenance={"run_log": parsed["path"], "runtime_profiles": runtime_profiles},
    )


def axis_b4_deployment_boundary(bundles: list[CaseBundle]) -> AxisResult:
    needles = (
        "validated operational decision support",
        "practitioner-proven",
        "proven practitioner utility",
    )
    hits: list[dict[str, str]] = []
    total = sum(len(bundle.syntheses) for bundle in bundles)
    for bundle in bundles:
        for synthesis in bundle.syntheses:
            text = str(synthesis.get("strategic_intelligence_report") or "").lower()
            for needle in needles:
                if needle in text:
                    hits.append(
                        {
                            "case_id": bundle.case_id,
                            "scenario_id": str(synthesis["scenario_id"]),
                            "pattern": needle,
                        }
                    )
    return AxisResult(
        experiment_id="B4",
        name="Deployment-claim boundary",
        status=EvaluationStatus.EVALUATED if not hits else EvaluationStatus.PARTIAL,
        scope=f"{total} final strategic reports",
        metrics=[
            _metric("deployment_overclaim_hit_count", len(hits), n=total),
            _metric("overclaim_hits", hits, n=len(hits)),
        ],
        findings=[
            "Without practitioner/deployment study data, the framework is bounded to evidence-grounded strategic decision support."
        ],
        limitations=(
            []
            if not hits
            else ["Potential deployment-claim overstatements require manual review before publication."]
        ),
    )


def axis_c1_jurisdiction(bundles: list[CaseBundle]) -> AxisResult:
    rows = _scenario_rows(bundles)
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_case[row["case_id"]].append(row)

    details: list[dict[str, Any]] = []
    stance_sensitive = 0
    evidence_sensitive = 0
    for case_id, case_rows in sorted(by_case.items()):
        case_rows.sort(key=lambda row: row["scenario_id"])
        lens_payload: dict[str, Any] = {}
        case_stance_sensitive = False
        case_evidence_sensitive = False
        for lens in LENSES:
            stances = [row["stances"][lens] for row in case_rows]
            evidence_sets = [row["evidence_ids_by_lens"][lens] for row in case_rows]
            stance_var = population_variance([float(value) for value in stances])
            evidence_overlap = pairwise_mean_jaccard(evidence_sets)
            lens_stance_sensitive = len(set(stances)) > 1
            lens_evidence_sensitive = evidence_overlap is not None and evidence_overlap < 1.0
            case_stance_sensitive |= lens_stance_sensitive
            case_evidence_sensitive |= lens_evidence_sensitive
            lens_payload[lens] = {
                "stances": stances,
                "stance_variance": stance_var,
                "pairwise_mean_evidence_jaccard": evidence_overlap,
                "stance_sensitive": lens_stance_sensitive,
                "evidence_sensitive": lens_evidence_sensitive,
            }
        stance_sensitive += int(case_stance_sensitive)
        evidence_sensitive += int(case_evidence_sensitive)
        details.append(
            {
                "case_id": case_id,
                "jurisdictions": [row["jurisdiction_code"] for row in case_rows],
                "lens_results": lens_payload,
                "any_stance_sensitivity": case_stance_sensitive,
                "any_evidence_sensitivity": case_evidence_sensitive,
            }
        )
    return AxisResult(
        experiment_id="C1",
        name="Jurisdiction sensitivity",
        status=EvaluationStatus.EVALUATED,
        scope=f"{len(by_case)} cases x KR/EU/US under fixed CIS IG2 reference profile",
        metrics=[
            _metric("cases_with_stance_sensitivity", stance_sensitive, n=len(by_case)),
            _metric("cases_with_evidence_sensitivity", evidence_sensitive, n=len(by_case)),
            _metric("case_details", details, n=len(details)),
        ],
        findings=[
            "Within-scenario D_lens is kept separate from cross-jurisdiction stance/evidence sensitivity."
        ],
        limitations=[
            "Three predeclared jurisdictions under one reference-enterprise profile do not establish generalizability beyond those frozen scenarios."
        ],
    )


def axis_c2_contextualized_path(
    bundles: list[CaseBundle],
    variants: list[VariantResultRecord],
) -> tuple[AxisResult, list[dict[str, Any]]]:
    return _variant_result(
        experiment_id="C2",
        name="Context-free vs contextualized decision-path comparison",
        axis_name="C2_CONTEXTUALIZED_PATH_COMPARISON",
        bundles=bundles,
        variants=variants,
    )


def axis_c3_scenario_provenance(bundles: list[CaseBundle]) -> AxisResult:
    scenario_versions = {str(bundle.stage3_context.get("scenario_set_version")) for bundle in bundles}
    manifest_hashes = {str(bundle.stage3_context.get("scenario_manifest_sha256")) for bundle in bundles}
    violations: list[str] = []
    per_case: list[dict[str, Any]] = []
    for bundle in bundles:
        objects = bundle.contextual_objects
        jurisdictions: list[str] = []
        actions: set[str] = set()
        profiles: set[str] = set()
        for obj in objects:
            context = obj.get("context_scenario") or {}
            deployment = obj.get("deployment_context") or {}
            jurisdictions.append(
                str(
                    context.get("jurisdiction_code")
                    or context.get("jurisdiction")
                    or obj.get("scenario_id")
                    or ""
                )
            )
            action = obj.get("candidate_action") or {}
            actions.add(
                str(
                    action.get("action_id")
                    or action.get("id")
                    or obj.get("pmt_or_mitigation_id")
                    or ""
                )
            )
            raw_profile = (
                context.get("reference_enterprise_profile")
                or deployment.get("reference_enterprise_profile")
                or "CIS_IG2"
            )
            if isinstance(raw_profile, dict):
                profile = str(raw_profile.get("value") or "")
            else:
                profile = str(raw_profile)
            profiles.add(profile)
        if (
            len(objects) != 3
            or set(jurisdictions) != {"KR", "EU", "US"}
            or len(actions) != 1
            or profiles != {"CIS_IG2"}
        ):
            violations.append(bundle.case_id)
        per_case.append(
            {
                "case_id": bundle.case_id,
                "scenario_count": len(objects),
                "jurisdictions": sorted(jurisdictions),
                "candidate_action_ids": sorted(actions),
                "reference_profiles": sorted(profiles),
            }
        )
    status = (
        EvaluationStatus.EVALUATED
        if not violations and len(scenario_versions) == 1 and len(manifest_hashes) == 1
        else EvaluationStatus.PARTIAL
    )
    return AxisResult(
        experiment_id="C3",
        name="Scenario-definition provenance",
        status=status,
        scope="main contextual scenario manifests",
        metrics=[
            _metric("scenario_set_version_count", len(scenario_versions), n=len(scenario_versions)),
            _metric("scenario_manifest_hash_count", len(manifest_hashes), n=len(manifest_hashes)),
            _metric("case_contract_details", per_case, n=len(per_case)),
            _metric("scenario_contract_violation_count", len(violations), n=len(per_case)),
        ],
        findings=[
            "Scenario definition, exact KR/EU/US jurisdiction set, fixed candidate action, and CIS_IG2 reference-enterprise profile are provenance-bound; scenario changes require a new version rather than silent reuse."
        ],
        limitations=([] if not violations else [f"Scenario contract violations: {violations}"]),
    )
