from __future__ import annotations

import copy
import re
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


STAGE1_SEMANTIC_VALIDATION_VERSION = "stage1-semantic-validation-v4"


class Stage1ValidationError(ValueError):
    """Raised when a Stage 1 CriticAssessment violates the fixed contract."""


class CriticType(StrEnum):
    ATTACK_FEASIBILITY = "attack_feasibility"
    DEFENSE_ROBUSTNESS = "defense_robustness"


class EvidenceSufficiency(StrEnum):
    SUFFICIENT = "SUFFICIENT"
    PARTIAL = "PARTIAL"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class ForecastRelation(StrEnum):
    SUPPORTS_FORECAST = "SUPPORTS_FORECAST"
    CHALLENGES_FORECAST = "CHALLENGES_FORECAST"
    NEUTRAL_CONTEXT = "NEUTRAL_CONTEXT"


class ForecastComponent(StrEnum):
    THREAT_TRAJECTORY = "THREAT_TRAJECTORY"
    PMT_TRAJECTORY = "PMT_TRAJECTORY"
    GAP_DIRECTION = "GAP_DIRECTION"
    OPERATIONAL_INTERPRETATION = "OPERATIONAL_INTERPRETATION"
    CONTEXT = "CONTEXT"


class StanceDecision(StrEnum):
    SUPPORT_DOMINATES = "SUPPORT_DOMINATES"
    CHALLENGE_DOMINATES = "CHALLENGE_DOMINATES"
    BALANCED = "BALANCED"
    NO_DIRECTIONAL_EVIDENCE = "NO_DIRECTIONAL_EVIDENCE"


class ClaimAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)
    forecast_component: ForecastComponent
    forecast_relation: ForecastRelation

    @model_validator(mode="after")
    def validate_evidence_ids(self) -> "ClaimAssessment":
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("evidence_ids must be unique within one ClaimAssessment")
        return self


class StanceBasis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    supporting_claim_ids: list[str] = Field(default_factory=list)
    challenging_claim_ids: list[str] = Field(default_factory=list)
    decision: StanceDecision
    rationale: str = Field(min_length=1)


class CriticAssessment(BaseModel):
    """Shared Stage 1 output contract for both independent forecast critics."""

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    critic_type: CriticType
    stance: Literal[-1, 0, 1]
    evidence_sufficiency: EvidenceSufficiency
    claims: list[ClaimAssessment] = Field(min_length=1)
    stance_basis: StanceBasis
    unresolved_questions: list[str] = Field(default_factory=list)
    unsupported_specificity_detected: bool

    @model_validator(mode="after")
    def validate_claim_ids(self) -> "CriticAssessment":
        claim_ids = [claim.claim_id for claim in self.claims]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("claim_id values must be unique within one CriticAssessment")

        claims_by_id = {claim.claim_id: claim for claim in self.claims}
        support_ids = self.stance_basis.supporting_claim_ids
        challenge_ids = self.stance_basis.challenging_claim_ids
        if len(support_ids) != len(set(support_ids)):
            raise ValueError("stance_basis.supporting_claim_ids must be unique")
        if len(challenge_ids) != len(set(challenge_ids)):
            raise ValueError("stance_basis.challenging_claim_ids must be unique")
        overlap = set(support_ids) & set(challenge_ids)
        if overlap:
            raise ValueError(f"The same claim cannot support and challenge the stance basis: {sorted(overlap)}")

        unknown_basis_ids = sorted((set(support_ids) | set(challenge_ids)) - set(claims_by_id))
        if unknown_basis_ids:
            raise ValueError(f"stance_basis references unknown claim IDs: {unknown_basis_ids}")
        wrong_support = sorted(
            claim_id
            for claim_id in support_ids
            if claims_by_id[claim_id].forecast_relation != ForecastRelation.SUPPORTS_FORECAST
        )
        if wrong_support:
            raise ValueError(f"stance_basis supporting claims must have SUPPORTS_FORECAST relation: {wrong_support}")
        wrong_challenge = sorted(
            claim_id
            for claim_id in challenge_ids
            if claims_by_id[claim_id].forecast_relation != ForecastRelation.CHALLENGES_FORECAST
        )
        if wrong_challenge:
            raise ValueError(
                f"stance_basis challenging claims must have CHALLENGES_FORECAST relation: {wrong_challenge}"
            )

        non_decisive_components = {
            ForecastComponent.OPERATIONAL_INTERPRETATION,
            ForecastComponent.CONTEXT,
        }
        non_decisive_basis = sorted(
            claim_id
            for claim_id in (*support_ids, *challenge_ids)
            if claims_by_id[claim_id].forecast_component in non_decisive_components
        )
        if non_decisive_basis:
            raise ValueError(
                "stance_basis may use only THREAT_TRAJECTORY, PMT_TRAJECTORY, or GAP_DIRECTION claims; "
                f"operational/context claims cannot be decisive by themselves: {non_decisive_basis}"
            )

        decisive_components = (
            {ForecastComponent.THREAT_TRAJECTORY, ForecastComponent.GAP_DIRECTION}
            if self.critic_type == CriticType.ATTACK_FEASIBILITY
            else {ForecastComponent.PMT_TRAJECTORY, ForecastComponent.GAP_DIRECTION}
        )
        directional_support = {
            claim.claim_id
            for claim in self.claims
            if claim.forecast_relation == ForecastRelation.SUPPORTS_FORECAST
            and claim.forecast_component in decisive_components
        }
        directional_challenge = {
            claim.claim_id
            for claim in self.claims
            if claim.forecast_relation == ForecastRelation.CHALLENGES_FORECAST
            and claim.forecast_component in decisive_components
        }
        decision = self.stance_basis.decision
        if decision == StanceDecision.SUPPORT_DOMINATES:
            if self.stance != 1 or not support_ids:
                raise ValueError("SUPPORT_DOMINATES requires stance=+1 and at least one supporting basis claim")
        elif decision == StanceDecision.CHALLENGE_DOMINATES:
            if self.stance != -1 or not challenge_ids:
                raise ValueError("CHALLENGE_DOMINATES requires stance=-1 and at least one challenging basis claim")
        elif decision == StanceDecision.BALANCED:
            if self.stance != 0 or not support_ids or not challenge_ids:
                raise ValueError("BALANCED requires stance=0 plus both supporting and challenging basis claims")
        elif decision == StanceDecision.NO_DIRECTIONAL_EVIDENCE:
            if self.stance != 0:
                raise ValueError("NO_DIRECTIONAL_EVIDENCE requires stance=0")
            if directional_support or directional_challenge:
                raise ValueError(
                    "NO_DIRECTIONAL_EVIDENCE is allowed only when no claim has a directional forecast relation"
                )
        return self


def build_constrained_critic_response_schema(
    *,
    evidence_pack: dict[str, Any],
    expected_critic_type: CriticType,
) -> dict[str, Any]:
    """Bind structured generation to the exact Stage 0 case and evidence-ID set."""

    case_id = str(evidence_pack.get("case_id") or "")
    if not case_id:
        raise Stage1ValidationError("Stage 0 EvidencePack is missing case_id")
    evidence_ids = sorted(
        {
            str(record.get("evidence_id"))
            for record in evidence_pack.get("evidence", [])
            if record.get("evidence_id")
        }
    )
    if not evidence_ids:
        raise Stage1ValidationError("Stage 0 EvidencePack contains no evidence IDs")

    schema = copy.deepcopy(CriticAssessment.model_json_schema())
    schema["properties"]["case_id"] = {"type": "string", "const": case_id}
    schema["properties"]["critic_type"] = {
        "type": "string",
        "const": expected_critic_type.value,
    }
    schema["$defs"]["ClaimAssessment"]["properties"]["evidence_ids"]["items"] = {
        "type": "string",
        "enum": evidence_ids,
    }
    return schema


def validate_critic_assessment(
    assessment: CriticAssessment | dict[str, Any],
    *,
    evidence_pack: dict[str, Any],
    expected_critic_type: CriticType,
) -> CriticAssessment:
    """Validate model output against the exact Stage 0 case and evidence IDs."""

    try:
        parsed = assessment if isinstance(assessment, CriticAssessment) else CriticAssessment.model_validate(assessment)
    except Exception as exc:  # Pydantic error details are preserved as the cause.
        raise Stage1ValidationError(f"Invalid CriticAssessment schema: {exc}") from exc

    expected_case_id = str(evidence_pack.get("case_id") or "")
    if not expected_case_id:
        raise Stage1ValidationError("Stage 0 EvidencePack is missing case_id")
    if parsed.case_id != expected_case_id:
        raise Stage1ValidationError(
            f"CriticAssessment case_id {parsed.case_id!r} does not match EvidencePack {expected_case_id!r}"
        )
    if parsed.critic_type != expected_critic_type:
        raise Stage1ValidationError(
            f"CriticAssessment critic_type {parsed.critic_type.value!r} does not match expected {expected_critic_type.value!r}"
        )

    decisive_components_by_critic = {
        CriticType.ATTACK_FEASIBILITY: {
            ForecastComponent.THREAT_TRAJECTORY,
            ForecastComponent.GAP_DIRECTION,
        },
        CriticType.DEFENSE_ROBUSTNESS: {
            ForecastComponent.PMT_TRAJECTORY,
            ForecastComponent.GAP_DIRECTION,
        },
    }
    allowed_decisive_components = decisive_components_by_critic[expected_critic_type]
    claims_by_id = {claim.claim_id: claim for claim in parsed.claims}
    validation_errors: list[str] = []
    non_role_basis = sorted(
        claim_id
        for claim_id in (
            *parsed.stance_basis.supporting_claim_ids,
            *parsed.stance_basis.challenging_claim_ids,
        )
        if claims_by_id[claim_id].forecast_component not in allowed_decisive_components
    )
    if non_role_basis:
        allowed = ", ".join(sorted(component.value for component in allowed_decisive_components))
        validation_errors.append(
            f"{expected_critic_type.value} stance_basis may use only role-relevant forecast components "
            f"({allowed}); invalid basis claim IDs: {non_role_basis}"
        )

    evidence_records = {
        str(record.get("evidence_id")): record
        for record in evidence_pack.get("evidence", [])
        if record.get("evidence_id")
    }
    known_evidence_ids = {
        evidence_id for evidence_id in evidence_records
    }
    referenced_ids = {
        evidence_id
        for claim in parsed.claims
        for evidence_id in claim.evidence_ids
    }
    unknown = sorted(referenced_ids - known_evidence_ids)
    if unknown:
        allowed = sorted(known_evidence_ids)
        validation_errors.append(
            "CriticAssessment references unknown Stage 0 evidence IDs: "
            f"{unknown}. Allowed evidence IDs for this case are: {allowed}"
        )

    forecast_summary = evidence_pack.get("forecast_summary") or {}
    threat_state = forecast_summary.get("threat_state") or {}
    pmt_state = forecast_summary.get("pmt_state") or {}
    threat_modality = str(threat_state.get("state_modality") or "")
    pmt_modality = str(pmt_state.get("state_modality") or "")
    expected_threat_id = str(evidence_pack.get("threat_id") or "")
    expected_pmt_id = str(evidence_pack.get("pmt_id") or "")

    def explicitly_reports_incident_count_direction(text: str) -> bool:
        """Require NoI direction to be about incident count/frequency, not another threat metric."""
        normalized = re.sub(r"\s+", " ", str(text).lower())
        patterns = (
            r"\b(?:number|count|frequency)\s+of\s+(?:[a-z0-9_-]+\s+){0,5}incidents?\b.{0,140}\b(?:increas\w*|decreas\w*|rose|risen|fell|fallen|grew|grown|declin\w*|dropp\w*|higher|lower|from|compared)\b",
            r"\bincidents?\s+(?:count|frequency)?\b.{0,80}\b(?:increas\w*|decreas\w*|rose|risen|fell|fallen|grew|grown|declin\w*|dropp\w*|higher|lower)\b",
            r"\b(?:increas\w*|decreas\w*|rise|rising|fall|falling|growth|decline|declining|drop|dropping|uptick|downturn)\s+(?:in\s+)?(?:the\s+)?(?:number|count|frequency)\s+of\s+(?:[a-z0-9_-]+\s+){0,5}incidents?\b",
        )
        return any(re.search(pattern, normalized) for pattern in patterns)

    def supports_direct_trajectory(
        record: dict[str, Any],
        *,
        entity: str,
        modality: str,
        claim_statement: str,
    ) -> bool:
        allowed_claim_types = {str(item) for item in record.get("allowed_claim_types", [])}
        if entity == "threat":
            if expected_threat_id and expected_threat_id not in {
                str(item) for item in record.get("threat_ids", [])
            }:
                return False
        else:
            if expected_pmt_id and expected_pmt_id not in {
                str(item) for item in record.get("pmt_ids", [])
            }:
                return False

        if modality == "NoI":
            # NoI is the paper's incident-frequency / incident-count state. A
            # directional threat report about attack size, severity, tactic
            # prevalence, malware-family share, or another metric is not a
            # direct NoI trajectory observation merely because the source is
            # allowed to report directional threat trends.
            return (
                "directional_threat_trend" in allowed_claim_types
                and explicitly_reports_incident_count_direction(record.get("content", ""))
                and explicitly_reports_incident_count_direction(claim_statement)
            )
        if modality == "NoP":
            # A bibliographic record proves that a publication exists; it does
            # not, by itself, establish an increase/decrease in publication
            # activity. Directional NoP critique therefore requires evidence
            # whose source contract explicitly represents a temporal
            # publication trend rather than static publication metadata.
            return "directional_publication_trend" in allowed_claim_types
        return False

    directional_relations = {
        ForecastRelation.SUPPORTS_FORECAST,
        ForecastRelation.CHALLENGES_FORECAST,
    }
    for claim in parsed.claims:
        if claim.forecast_relation not in directional_relations:
            continue
        cited_records = [
            evidence_records[evidence_id]
            for evidence_id in claim.evidence_ids
            if evidence_id in evidence_records
        ]

        if claim.forecast_component == ForecastComponent.THREAT_TRAJECTORY and threat_modality:
            if not any(
                supports_direct_trajectory(
                    record,
                    entity="threat",
                    modality=threat_modality,
                    claim_statement=claim.statement,
                )
                for record in cited_records
            ):
                validation_errors.append(
                    f"Claim {claim.claim_id!r} is labeled THREAT_TRAJECTORY but its cited evidence does not "
                    f"directly ground the Threat {threat_modality or 'state'} trajectory. For NoI, the cited record "
                    "and claim must explicitly report a temporal increase/decrease in incident count or frequency; "
                    "attack magnitude, severity, tactic prevalence, malware-family share, or a percentage/share of "
                    "intrusions is not a direct NoI trajectory observation. The source contract must also allow "
                    "directional_threat_trend and be tagged to this threat. For NoP, "
                    "use evidence whose source contract explicitly allows directional_publication_trend and is "
                    "tagged to this threat. Static publication_metadata establishes publication existence/topic, "
                    "not an increase or decrease in NoP. Otherwise reclassify the claim as "
                    "OPERATIONAL_INTERPRETATION or CONTEXT and remove it from stance_basis if necessary."
                )
        elif claim.forecast_component == ForecastComponent.PMT_TRAJECTORY and pmt_modality:
            if not any(
                supports_direct_trajectory(
                    record,
                    entity="pmt",
                    modality=pmt_modality,
                    claim_statement=claim.statement,
                )
                for record in cited_records
            ):
                validation_errors.append(
                    f"Claim {claim.claim_id!r} is labeled PMT_TRAJECTORY but its cited evidence does not directly "
                    f"ground the PMT {pmt_modality or 'state'} trajectory. For NoP, use evidence whose source "
                    "contract explicitly allows directional_publication_trend and is tagged to this PMT. Static "
                    "publication_metadata establishes publication existence/topic, not an increase or decrease in "
                    "NoP; control existence, deployment guidance, implementation procedures, maturity, or operational "
                    "effectiveness likewise belong under OPERATIONAL_INTERPRETATION/CONTEXT instead."
                )
        elif (
            claim.forecast_component == ForecastComponent.GAP_DIRECTION
            and threat_modality
            and pmt_modality
        ):
            has_threat_side = any(
                supports_direct_trajectory(
                    record,
                    entity="threat",
                    modality=threat_modality,
                    claim_statement=claim.statement,
                )
                for record in cited_records
            )
            has_pmt_side = any(
                supports_direct_trajectory(
                    record,
                    entity="pmt",
                    modality=pmt_modality,
                    claim_statement=claim.statement,
                )
                for record in cited_records
            )
            if not (has_threat_side and has_pmt_side):
                validation_errors.append(
                    f"Claim {claim.claim_id!r} is labeled GAP_DIRECTION but does not cite direct evidence for both "
                    f"forecast-state sides (Threat {threat_modality or 'state'} and PMT {pmt_modality or 'state'}). "
                    "A one-sided operational weakness/strength cannot directly ground the Threat-PMT gap direction; "
                    "reclassify it as OPERATIONAL_INTERPRETATION/CONTEXT unless both sides are directly evidenced."
                )

    if validation_errors:
        raise Stage1ValidationError(" | ".join(validation_errors))

    return parsed
