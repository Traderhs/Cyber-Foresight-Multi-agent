from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Stage1ValidationError(ValueError):
    """Raised when a Stage 1 CriticAssessment violates the fixed contract."""


class CriticType(StrEnum):
    ATTACK_FEASIBILITY = "attack_feasibility"
    DEFENSE_ROBUSTNESS = "defense_robustness"


class EvidenceSufficiency(StrEnum):
    SUFFICIENT = "SUFFICIENT"
    PARTIAL = "PARTIAL"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class ClaimStatus(StrEnum):
    SUPPORTED = "SUPPORTED"
    CHALLENGED = "CHALLENGED"
    MIXED = "MIXED"
    UNRESOLVED = "UNRESOLVED"


class ClaimAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    contradicting_evidence_ids: list[str] = Field(default_factory=list)
    status: ClaimStatus

    @model_validator(mode="after")
    def validate_evidence_roles(self) -> "ClaimAssessment":
        supporting = set(self.supporting_evidence_ids)
        contradicting = set(self.contradicting_evidence_ids)
        overlap = supporting & contradicting
        if overlap:
            raise ValueError(f"Evidence IDs cannot support and contradict the same claim: {sorted(overlap)}")
        if self.status == ClaimStatus.SUPPORTED and not supporting:
            raise ValueError("SUPPORTED claim requires at least one supporting evidence ID")
        if self.status == ClaimStatus.CHALLENGED and not contradicting:
            raise ValueError("CHALLENGED claim requires at least one contradicting evidence ID")
        if self.status == ClaimStatus.MIXED and (not supporting or not contradicting):
            raise ValueError("MIXED claim requires both supporting and contradicting evidence IDs")
        return self


class CriticAssessment(BaseModel):
    """Shared Stage 1 output contract for both independent forecast critics."""

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    critic_type: CriticType
    stance: Literal[-1, 0, 1]
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_sufficiency: EvidenceSufficiency
    claims: list[ClaimAssessment] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    unsupported_specificity_detected: bool

    @model_validator(mode="after")
    def validate_claim_ids(self) -> "CriticAssessment":
        claim_ids = [claim.claim_id for claim in self.claims]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("claim_id values must be unique within one CriticAssessment")
        return self


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

    known_evidence_ids = {
        str(record.get("evidence_id"))
        for record in evidence_pack.get("evidence", [])
        if record.get("evidence_id")
    }
    referenced_ids = {
        evidence_id
        for claim in parsed.claims
        for evidence_id in (*claim.supporting_evidence_ids, *claim.contradicting_evidence_ids)
    }
    unknown = sorted(referenced_ids - known_evidence_ids)
    if unknown:
        raise Stage1ValidationError(f"CriticAssessment references unknown Stage 0 evidence IDs: {unknown}")

    return parsed
