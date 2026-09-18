from __future__ import annotations

import hashlib
import json
from typing import Any

from Stage1.schema import ForecastRelation
from Stage2.schema import MediatorAdjudicationOutcome, Stage2DebateResult
from Stage3.schema import (
    CandidateActionRef,
    CandidateActionSource,
    DecisionObject,
    DeploymentContext,
    ExcludedAction,
    RequiredRevisionRef,
    STAGE3_SELECTION_RULE_VERSION,
    Stage2CritiqueSummary,
    Stage3DecisionBundle,
    Stage3ValidationError,
    TimingWindow,
)


_DEPLOYMENT_CONTEXT_FIELDS = (
    "region",
    "organization_type",
    "infrastructure_context",
    "budget_context",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _short_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()[:16]


def _sorted_unique(values: list[str]) -> list[str]:
    return sorted(set(values))


def _forecast_evidence_ids(post_assessment: dict[str, Any], relation: ForecastRelation) -> list[str]:
    return _sorted_unique(
        [
            evidence_id
            for claim in post_assessment.get("claims", [])
            if claim.get("forecast_relation") == relation.value
            for evidence_id in claim.get("evidence_ids", [])
        ]
    )


def _matched_claim_ids(items: list[dict[str, Any]]) -> list[str]:
    return _sorted_unique(
        [
            claim_id
            for item in items
            for claim_id in [
                *item.get("attack_claim_ids", []),
                *item.get("defense_claim_ids", []),
            ]
        ]
    )


def build_stage3_bundle(
    *,
    evidence_pack: dict[str, Any],
    stage0_config: dict[str, Any],
    stage2_result: dict[str, Any],
) -> Stage3DecisionBundle:
    """Build the deterministic Stage 3 evaluation unit for the current forecast pair."""

    if not stage0_config:
        raise Stage3ValidationError("Stage 3 requires the explicit Stage 0 case configuration")
    if not stage2_result:
        raise Stage3ValidationError("Stage 3 requires a completed Stage 2 result")

    # Stage 2 semantic validation already happened before the immutable Stage 2
    # artifact was frozen. Stage 3 verifies that exact artifact at the node
    # boundary and only re-validates the structured result shape here rather
    # than re-judging Stage 2 evidence semantics.
    parsed_stage2 = Stage2DebateResult.model_validate(stage2_result)
    stage2 = parsed_stage2.model_dump(mode="json")

    case_id = str(evidence_pack.get("case_id") or "")
    threat_id = str(evidence_pack.get("threat_id") or "")
    pmt_id = str(evidence_pack.get("pmt_id") or "")
    threat_name = str(stage0_config.get("threat") or threat_id)
    pmt_name = str(stage0_config.get("pmt") or pmt_id)
    if not all((case_id, threat_id, pmt_id, threat_name, pmt_name)):
        raise Stage3ValidationError("Stage 3 case identity is incomplete")
    for config_key, pack_key in (
        ("case_id", "case_id"),
        ("snapshot_id", "source_snapshot_id"),
        ("analysis_cutoff_date", "analysis_cutoff_date"),
        ("evaluation_mode", "evaluation_mode"),
    ):
        configured = stage0_config.get(config_key)
        packed = evidence_pack.get(pack_key)
        if configured not in (None, "") and packed not in (None, "") and str(configured) != str(packed):
            raise Stage3ValidationError(
                f"Stage 3 Stage0 config mismatch for {config_key}: config={configured!r}, pack={packed!r}"
            )
    if stage2["case_id"] != case_id:
        raise Stage3ValidationError("Stage 2/Stage 0 case_id mismatch at Stage 3")

    evaluation_evidence_ids = _sorted_unique(
        [str(record["evidence_id"]) for record in evidence_pack.get("evidence", []) if record.get("evidence_id")]
    )
    if not evaluation_evidence_ids:
        raise Stage3ValidationError("Stage 3 requires a non-empty frozen Stage 0 evaluation evidence set")

    candidate_source_evidence_ids = _sorted_unique(
        [
            str(record["evidence_id"])
            for record in evidence_pack.get("evidence", [])
            if pmt_id in {str(item) for item in record.get("pmt_ids", [])}
        ]
    )
    candidate = CandidateActionRef(
        action_id=pmt_id,
        action_name=pmt_name,
        source=CandidateActionSource.B_MTGNN_CASE_PMT,
        source_evidence_ids=candidate_source_evidence_ids,
    )

    evidence_tagged_other_pmts = sorted(
        {
            str(tagged_pmt)
            for record in evidence_pack.get("evidence", [])
            for tagged_pmt in record.get("pmt_ids", [])
            if str(tagged_pmt) != pmt_id
        }
    )
    excluded = [
        ExcludedAction(
            action_id=action_id,
            reason=(
                "Excluded by case-forecast-pmt-only-v1: the main Stage 3 object evaluates the PMT in the "
                "predeclared B-MTGNN Threat-PMT forecast pair. Evidence-tagged alternative PMTs are not silently "
                "promoted into the main action set; they are reserved for an explicit action-set sensitivity run."
            ),
        )
        for action_id in evidence_tagged_other_pmts
    ]

    final_adjudications = stage2["final_adjudications"]
    revision_outcomes = {
        MediatorAdjudicationOutcome.REVISE_ATTACK.value,
        MediatorAdjudicationOutcome.REVISE_DEFENSE.value,
        MediatorAdjudicationOutcome.REVISE_BOTH.value,
    }
    required_revisions = [
        RequiredRevisionRef(
            adjudication_id=item["adjudication_id"],
            outcome=item["outcome"],
            attack_claim_ids=item.get("attack_claim_ids", []),
            defense_claim_ids=item.get("defense_claim_ids", []),
            required_revision=item["required_revision"],
        )
        for item in final_adjudications
        if item["outcome"] in revision_outcomes and item.get("required_revision")
    ]

    critique_summary = Stage2CritiqueSummary(
        attack_post_stance=int(stage2["attack_post_assessment"]["stance"]),
        defense_post_stance=int(stage2["defense_post_assessment"]["stance"]),
        final_mediator_adjudications=final_adjudications,
        matched_resolved_claim_ids=_matched_claim_ids(stage2.get("resolved_claims", [])),
        matched_unresolved_claim_ids=_matched_claim_ids(stage2.get("unresolved_claims", [])),
        required_revisions=required_revisions,
        attack_post_unresolved_questions=_sorted_unique(
            [str(item) for item in stage2["attack_post_assessment"].get("unresolved_questions", [])]
        ),
        defense_post_unresolved_questions=_sorted_unique(
            [str(item) for item in stage2["defense_post_assessment"].get("unresolved_questions", [])]
        ),
    )

    forecast_summary = evidence_pack.get("forecast_summary") or {}
    required_forecast_fields = (
        "threat_state",
        "pmt_state",
        "gap_by_year",
        "gap_slope_per_year",
        "gap_direction",
        "gap_semantics",
        "predictive_uncertainty",
    )
    missing_forecast_fields = [field for field in required_forecast_fields if forecast_summary.get(field) is None]
    if missing_forecast_fields:
        raise Stage3ValidationError(
            f"Stage 3 frozen forecast summary is missing required fields: {missing_forecast_fields}"
        )
    gap_by_year = forecast_summary.get("gap_by_year") or {}
    years = sorted(int(year) for year in gap_by_year)
    if not years:
        raise Stage3ValidationError("Stage 3 requires the frozen forecast gap timing window")

    deployment_values = {
        field: (stage0_config.get(field) if stage0_config.get(field) not in (None, "") else None)
        for field in _DEPLOYMENT_CONTEXT_FIELDS
    }
    unknown_context_fields = [
        field for field, value in deployment_values.items() if value in (None, "")
    ]
    known_constraints = [
        "Candidate action is frozen before downstream lens evaluation under case-forecast-pmt-only-v1.",
        "Stage 2 adjudications are reasoning provenance and are not promoted to external evidence.",
    ]
    if unknown_context_fields:
        known_constraints.append(
            "Organization/deployment-specific feasibility must remain conditional where deployment context is unknown."
        )

    supporting_evidence_ids = _sorted_unique(
        _forecast_evidence_ids(stage2["attack_post_assessment"], ForecastRelation.SUPPORTS_FORECAST)
        + _forecast_evidence_ids(stage2["defense_post_assessment"], ForecastRelation.SUPPORTS_FORECAST)
    )
    contradictory_evidence_ids = _sorted_unique(
        _forecast_evidence_ids(stage2["attack_post_assessment"], ForecastRelation.CHALLENGES_FORECAST)
        + _forecast_evidence_ids(stage2["defense_post_assessment"], ForecastRelation.CHALLENGES_FORECAST)
    )

    forecast_interpretation = {
        "threat_state": forecast_summary.get("threat_state"),
        "pmt_state": forecast_summary.get("pmt_state"),
        "gap_by_year": gap_by_year,
        "gap_slope_per_year": forecast_summary.get("gap_slope_per_year"),
        "gap_direction": forecast_summary.get("gap_direction"),
        "gap_semantics": forecast_summary.get("gap_semantics"),
        "gap_direction_semantics": forecast_summary.get("gap_direction_semantics"),
    }
    decision_object_identity_payload = {
        "case_id": case_id,
        "action_id": pmt_id,
        "selection_rule": STAGE3_SELECTION_RULE_VERSION,
        "forecast_interpretation": forecast_interpretation,
        "stage2_result": stage2,
        "deployment_context": deployment_values,
        "evaluation_evidence_ids": evaluation_evidence_ids,
    }
    decision_object_id = (
        f"decision__{case_id}__{pmt_id.lower()}__{_short_hash(decision_object_identity_payload)}"
    )

    decision_object = DecisionObject(
        decision_object_id=decision_object_id,
        case_id=case_id,
        threat=threat_name,
        threat_id=threat_id,
        pmt_or_mitigation=pmt_name,
        pmt_or_mitigation_id=pmt_id,
        forecast_interpretation=forecast_interpretation,
        stage2_critique_summary=critique_summary,
        candidate_action=candidate,
        candidate_action_source=candidate.source,
        candidate_selection_rule=STAGE3_SELECTION_RULE_VERSION,
        intended_goal=(
            f"Evaluate the feasibility and decision relevance of {pmt_name} for the frozen {threat_name} "
            "forecast pair without changing the action set after observing downstream lens outputs."
        ),
        timing_window=TimingWindow(start_year=years[0], end_year=years[-1]),
        deployment_context=DeploymentContext(**deployment_values),
        known_constraints=known_constraints,
        unknown_context_fields=unknown_context_fields,
        evaluation_evidence_ids=evaluation_evidence_ids,
        supporting_evidence_ids=supporting_evidence_ids,
        contradictory_evidence_ids=contradictory_evidence_ids,
        unresolved_evidence_gaps=stage2.get("evidence_gaps", []),
        predictive_uncertainty=forecast_summary["predictive_uncertainty"],
    )

    return Stage3DecisionBundle(
        case_id=case_id,
        candidate_selection_rule=STAGE3_SELECTION_RULE_VERSION,
        eligible_action_set=[candidate],
        excluded_actions_and_reason=excluded,
        decision_objects=[decision_object],
    )
