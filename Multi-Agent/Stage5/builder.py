from __future__ import annotations

import json
from statistics import pvariance
from typing import Any

from Stage2.schema import Stage2DebateResult
from Stage3.context_schema import ContextualDecisionObject, Stage3ContextualDecisionBundle
from Stage4.context_schema import contextual_general_evidence_signature
from Stage4.schema import (
    LensAssessment,
    LensClaimScope,
    LensClaimStatus,
    LensEvidenceSufficiency,
    LensType,
    Stage4EvaluationBundle,
)
from Stage5.schema import (
    STAGE5_DIAGNOSTIC_CONTRACT_VERSION,
    AgreementClass,
    ClaimEvidenceAuditStatus,
    CritiqueDisagreementDiagnostic,
    DirectionalClaimSignature,
    JurisdictionSensitivitySummary,
    LensJurisdictionSensitivity,
    ScenarioLensDiagnostic,
    Stage5DiagnosticBundle,
    Stage5ValidationError,
    SystemEvidenceDiagnostic,
)


_LENS_FIELDS = {
    LensType.TECHNICAL_FEASIBILITY: "technical_feasibility",
    LensType.INSTITUTIONAL_REGIONAL: "institutional_regional",
    LensType.FINANCIAL_ADOPTION: "financial_adoption",
}

_GROUNDED_SUFFICIENCY = {
    LensEvidenceSufficiency.SUFFICIENT,
    LensEvidenceSufficiency.PARTIAL,
}


def _variance(values: list[int]) -> float:
    return round(float(pvariance(values)), 6)


def _assessment(evaluation: Any, lens_type: LensType) -> LensAssessment:
    return getattr(evaluation, _LENS_FIELDS[lens_type])


def _directional_claims(assessment: LensAssessment) -> list[Any]:
    return [
        claim
        for claim in assessment.claims
        if claim.status
        in {
            LensClaimStatus.SUPPORTS_FEASIBILITY,
            LensClaimStatus.CHALLENGES_FEASIBILITY,
        }
    ]


def _scenario_diagnostic(
    *,
    obj: ContextualDecisionObject,
    evaluation: Any,
) -> ScenarioLensDiagnostic:
    assessments = {lens: _assessment(evaluation, lens) for lens in LensType}
    stances = [assessments[lens].stance for lens in LensType]
    sufficiency = {
        lens: assessments[lens].evidence_sufficiency
        for lens in LensType
    }
    directional_claims = {
        lens: _directional_claims(assessments[lens])
        for lens in LensType
    }
    sufficient_or_partial = sum(
        value in _GROUNDED_SUFFICIENCY for value in sufficiency.values()
    )
    directional_lens_count = sum(bool(items) for items in directional_claims.values())
    joint_abstention = (
        all(stance == 0 for stance in stances)
        and sufficient_or_partial == 0
        and directional_lens_count == 0
    )
    d_lens = _variance(stances)
    d_lens_evaluable = sufficient_or_partial == 3
    if d_lens == 0:
        agreement_class = (
            AgreementClass.GROUNDED_AGREEMENT
            if d_lens_evaluable
            else AgreementClass.UNDER_INFORMED_CONVERGENCE
        )
    else:
        agreement_class = (
            AgreementClass.SUBSTANTIVE_DISAGREEMENT
            if d_lens_evaluable
            else AgreementClass.EVIDENCE_LIMITED_DISAGREEMENT
        )

    def _count(lens: LensType, status: LensClaimStatus) -> int:
        return sum(claim.status == status for claim in assessments[lens].claims)

    return ScenarioLensDiagnostic(
        decision_object_id=obj.decision_object_id,
        scenario_id=obj.scenario_id,
        jurisdiction_code=obj.context_scenario.jurisdiction_code,
        stance_vector={lens.value: assessments[lens].stance for lens in LensType},
        evidence_sufficiency_by_lens={
            lens.value: assessments[lens].evidence_sufficiency for lens in LensType
        },
        d_lens=d_lens,
        d_lens_evaluable=d_lens_evaluable,
        joint_abstention=joint_abstention,
        nonzero_stance_lens_count=sum(stance != 0 for stance in stances),
        directional_lens_count=directional_lens_count,
        sufficient_or_partial_lens_count=sufficient_or_partial,
        agreement_class=agreement_class,
        support_claim_count_by_lens={
            lens.value: _count(lens, LensClaimStatus.SUPPORTS_FEASIBILITY)
            for lens in LensType
        },
        challenge_claim_count_by_lens={
            lens.value: _count(lens, LensClaimStatus.CHALLENGES_FEASIBILITY)
            for lens in LensType
        },
        cited_evidence_ids_by_lens={
            lens.value: sorted(
                {
                    evidence_id
                    for claim in assessments[lens].claims
                    for evidence_id in claim.evidence_ids
                }
            )
            for lens in LensType
        },
        deployment_specific_directional_claim_count_by_lens={
            lens.value: sum(
                claim.scope == LensClaimScope.DEPLOYMENT_SPECIFIC
                for claim in directional_claims[lens]
            )
            for lens in LensType
        },
    )


def _signature(assessment: LensAssessment) -> list[DirectionalClaimSignature]:
    signatures = [
        DirectionalClaimSignature(
            status=claim.status.value,
            scope=claim.scope.value,
            evidence_ids=sorted(set(claim.evidence_ids)),
        )
        for claim in _directional_claims(assessment)
    ]
    return sorted(
        signatures,
        key=lambda item: (item.status, item.scope, tuple(item.evidence_ids)),
    )


def _dict_values_changed(values: dict[str, Any]) -> bool:
    normalized = [
        json.dumps(values[key], ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
        for key in ("KR", "EU", "US")
    ]
    return len(set(normalized)) > 1


def _jurisdiction_sensitivity(
    *,
    objects: list[ContextualDecisionObject],
    evaluations_by_id: dict[str, Any],
) -> JurisdictionSensitivitySummary:
    objects_by_jurisdiction = {
        obj.context_scenario.jurisdiction_code: obj
        for obj in objects
    }
    if set(objects_by_jurisdiction) != {"KR", "EU", "US"}:
        raise Stage5ValidationError("Stage 5 jurisdiction diagnostics require exactly KR/EU/US")

    by_lens: dict[str, LensJurisdictionSensitivity] = {}
    for lens in LensType:
        stance_by_jurisdiction: dict[str, int] = {}
        available_relevant: dict[str, list[str]] = {}
        cited_directional: dict[str, list[str]] = {}
        jurisdiction_specific_available: dict[str, list[str]] = {}
        jurisdiction_specific_used: dict[str, list[str]] = {}
        signatures: dict[str, list[DirectionalClaimSignature]] = {}
        deployment_specific_counts: dict[str, int] = {}

        for jurisdiction in ("KR", "EU", "US"):
            obj = objects_by_jurisdiction[jurisdiction]
            evaluation = evaluations_by_id[obj.decision_object_id]
            assessment = _assessment(evaluation, lens)
            directional = _directional_claims(assessment)
            cited_ids = sorted(
                {
                    evidence_id
                    for claim in directional
                    for evidence_id in claim.evidence_ids
                }
            )
            regulatory_ids = (
                sorted(
                    evidence_id
                    for evidence_id, slots in obj.decision_evidence_pack.slot_coverage.items()
                    if "regulatory_applicability" in set(slots)
                )
                if lens == LensType.INSTITUTIONAL_REGIONAL
                else []
            )
            stance_by_jurisdiction[jurisdiction] = assessment.stance
            available_relevant[jurisdiction] = list(
                contextual_general_evidence_signature(obj, lens)
            )
            cited_directional[jurisdiction] = cited_ids
            jurisdiction_specific_available[jurisdiction] = regulatory_ids
            jurisdiction_specific_used[jurisdiction] = sorted(set(cited_ids) & set(regulatory_ids))
            signatures[jurisdiction] = _signature(assessment)
            deployment_specific_counts[jurisdiction] = sum(
                claim.scope == LensClaimScope.DEPLOYMENT_SPECIFIC for claim in directional
            )

        available_changed = _dict_values_changed(available_relevant)
        cited_changed = _dict_values_changed(cited_directional)
        signature_changed = _dict_values_changed(
            {
                key: [item.model_dump(mode="json") for item in value]
                for key, value in signatures.items()
            }
        )
        deployment_count_changed = _dict_values_changed(deployment_specific_counts)
        jurisdiction_specific_available_changed = _dict_values_changed(
            jurisdiction_specific_available
        )
        jurisdiction_specific_used_changed = _dict_values_changed(jurisdiction_specific_used)
        sources: list[str] = []
        if available_changed:
            sources.append("AVAILABLE_LENS_RELEVANT_EVIDENCE")
        if cited_changed:
            sources.append("CITED_DIRECTIONAL_EVIDENCE")
        if signature_changed:
            sources.append("DIRECTIONAL_CLAIM_SIGNATURE")
        if deployment_count_changed:
            sources.append("DEPLOYMENT_SPECIFIC_DIRECTIONAL_CLAIM_COUNT")
        if jurisdiction_specific_available_changed:
            sources.append("JURISDICTION_SPECIFIC_EVIDENCE_AVAILABLE")
        if jurisdiction_specific_used_changed:
            sources.append("JURISDICTION_SPECIFIC_EVIDENCE_USED")

        stances = [stance_by_jurisdiction[key] for key in ("KR", "EU", "US")]
        evidence_sensitive = bool(sources)
        diagnostic = LensJurisdictionSensitivity(
            lens_type=lens,
            stance_by_jurisdiction=stance_by_jurisdiction,
            stance_variance=_variance(stances),
            jurisdiction_stance_sensitivity=len(set(stances)) > 1,
            available_lens_relevant_evidence_ids_by_jurisdiction=available_relevant,
            cited_directional_evidence_ids_by_jurisdiction=cited_directional,
            jurisdiction_specific_available_evidence_ids_by_jurisdiction=(
                jurisdiction_specific_available
            ),
            jurisdiction_specific_used_evidence_ids_by_jurisdiction=(
                jurisdiction_specific_used
            ),
            directional_claim_signatures_by_jurisdiction=signatures,
            deployment_specific_directional_claim_count_by_jurisdiction=(
                deployment_specific_counts
            ),
            available_lens_relevant_evidence_changed=available_changed,
            cited_directional_evidence_changed=cited_changed,
            directional_claim_signature_changed=signature_changed,
            deployment_specific_directional_claim_count_changed=deployment_count_changed,
            jurisdiction_evidence_sensitivity=evidence_sensitive,
            sensitivity_sources=sorted(sources),
        )
        by_lens[lens.value] = diagnostic

    stance_sensitive = sorted(
        lens_name
        for lens_name, item in by_lens.items()
        if item.jurisdiction_stance_sensitivity
    )
    evidence_sensitive = sorted(
        lens_name
        for lens_name, item in by_lens.items()
        if item.jurisdiction_evidence_sensitivity
    )
    return JurisdictionSensitivitySummary(
        by_lens=by_lens,
        any_stance_sensitivity=bool(stance_sensitive),
        any_evidence_sensitivity=bool(evidence_sensitive),
        stance_sensitive_lenses=stance_sensitive,
        evidence_sensitive_lenses=evidence_sensitive,
    )


def _critique_diagnostic(stage2: Stage2DebateResult) -> CritiqueDisagreementDiagnostic:
    pre = [stage2.attack_pre_assessment.stance, stage2.defense_pre_assessment.stance]
    post = [stage2.attack_post_assessment.stance, stage2.defense_post_assessment.stance]
    d_pre = _variance(pre)
    d_post = _variance(post)
    return CritiqueDisagreementDiagnostic(
        attack_pre_stance=pre[0],
        defense_pre_stance=pre[1],
        d_critique_pre=d_pre,
        attack_post_stance=post[0],
        defense_post_stance=post[1],
        d_critique_post=d_post,
        d_critique_change=round(d_post - d_pre, 6),
        unresolved_claim_count=len(stage2.unresolved_claims),
        evidence_gap_count=len(stage2.evidence_gaps),
    )


def _system_evidence_diagnostic(
    *,
    evidence_pack: dict[str, Any],
    objects_by_id: dict[str, ContextualDecisionObject],
    stage4: Stage4EvaluationBundle,
) -> SystemEvidenceDiagnostic:
    metadata = evidence_pack.get("retrieval_metadata") or {}
    required = sorted(set(metadata.get("required_evidence_slots") or []))
    covered = sorted(set(metadata.get("covered_evidence_slots") or []))
    missing = sorted(set(metadata.get("missing_evidence_slots") or []))
    ratio = 1.0 if not required else round(len(covered) / len(required), 6)

    sufficiency_counts = {item.value: 0 for item in LensEvidenceSufficiency}
    claim_count = 0
    directional_count = 0
    reference_total = 0
    reference_valid = 0
    for evaluation in stage4.evaluations:
        obj = objects_by_id[evaluation.decision_object_id]
        allowed = set(obj.evaluation_evidence_ids)
        for lens in LensType:
            assessment = _assessment(evaluation, lens)
            sufficiency_counts[assessment.evidence_sufficiency.value] += 1
            claim_count += len(assessment.claims)
            directional_count += len(_directional_claims(assessment))
            for claim in assessment.claims:
                for evidence_id in claim.evidence_ids:
                    reference_total += 1
                    reference_valid += evidence_id in allowed

    pass_rate = 1.0 if reference_total == 0 else round(reference_valid / reference_total, 6)
    return SystemEvidenceDiagnostic(
        required_evidence_slots=required,
        covered_evidence_slots=covered,
        missing_evidence_slots=missing,
        evidence_slot_coverage_ratio=ratio,
        source_family_count=int(metadata.get("source_family_count") or 0),
        independent_evidence_chain_count=int(
            metadata.get("independent_evidence_chain_count") or 0
        ),
        temporal_violation_count=int(metadata.get("temporal_violation_count") or 0),
        source_contract_violation_count=int(
            metadata.get("source_contract_violation_count") or 0
        ),
        stage0_evidence_sufficiency=str(metadata.get("evidence_sufficiency") or "UNKNOWN"),
        stage4_sufficiency_counts=sufficiency_counts,
        stage4_claim_count=claim_count,
        stage4_directional_claim_count=directional_count,
        claim_evidence_reference_pass_rate=pass_rate,
        claim_evidence_audit_pass_rate=None,
        claim_evidence_audit_status=(
            ClaimEvidenceAuditStatus.DEFERRED_TO_STAGE7_SEMANTIC_GROUNDING_AUDIT
        ),
    )


def build_stage5_diagnostic_bundle(
    *,
    evidence_pack: dict[str, Any],
    stage2_result: Stage2DebateResult | dict[str, Any],
    stage3_context_bundle: Stage3ContextualDecisionBundle | dict[str, Any],
    stage4_context_bundle: Stage4EvaluationBundle | dict[str, Any],
) -> Stage5DiagnosticBundle:
    try:
        stage2 = (
            stage2_result
            if isinstance(stage2_result, Stage2DebateResult)
            else Stage2DebateResult.model_validate(stage2_result)
        )
        stage3 = (
            stage3_context_bundle
            if isinstance(stage3_context_bundle, Stage3ContextualDecisionBundle)
            else Stage3ContextualDecisionBundle.model_validate(stage3_context_bundle)
        )
        stage4 = (
            stage4_context_bundle
            if isinstance(stage4_context_bundle, Stage4EvaluationBundle)
            else Stage4EvaluationBundle.model_validate(stage4_context_bundle)
        )
    except Exception as exc:
        raise Stage5ValidationError(f"Invalid upstream Stage 5 input: {exc}") from exc

    case_id = str(evidence_pack.get("case_id") or "")
    if not case_id or len({case_id, stage2.case_id, stage3.case_id, stage4.case_id}) != 1:
        raise Stage5ValidationError("Stage 5 upstream case_id mismatch")

    objects = stage3.contextual_decision_objects
    objects_by_id = {obj.decision_object_id: obj for obj in objects}
    evaluations_by_id = {item.decision_object_id: item for item in stage4.evaluations}
    if set(objects_by_id) != set(evaluations_by_id):
        raise Stage5ValidationError("Stage 5 requires exact Stage 3/4 decision-object alignment")

    scenario_diagnostics = [
        _scenario_diagnostic(
            obj=obj,
            evaluation=evaluations_by_id[obj.decision_object_id],
        )
        for obj in objects
    ]
    scenario_diagnostics.sort(key=lambda item: {"KR": 0, "EU": 1, "US": 2}[item.jurisdiction_code])

    bundle = Stage5DiagnosticBundle(
        case_id=case_id,
        diagnostic_contract_version=STAGE5_DIAGNOSTIC_CONTRACT_VERSION,
        scenario_diagnostics=scenario_diagnostics,
        critique_diagnostic=_critique_diagnostic(stage2),
        system_evidence_diagnostic=_system_evidence_diagnostic(
            evidence_pack=evidence_pack,
            objects_by_id=objects_by_id,
            stage4=stage4,
        ),
        jurisdiction_sensitivity=_jurisdiction_sensitivity(
            objects=objects,
            evaluations_by_id=evaluations_by_id,
        ),
        model_reported_confidence_used=False,
        predictive_uncertainty_claimed_from_d_lens=False,
    )
    return bundle
