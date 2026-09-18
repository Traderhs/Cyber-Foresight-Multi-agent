from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from Stage4.schema import LensAssessment, LensType
from Stage7.config import STAGE7_VARIANT_RECORD_VERSION


class EvaluationStatus(StrEnum):
    EVALUATED = "EVALUATED"
    PARTIAL = "PARTIAL"
    NOT_EVALUATED = "NOT_EVALUATED"
    ERROR = "ERROR"


class GroundingVerdict(StrEnum):
    DIRECT_SUPPORT = "DIRECT_SUPPORT"
    PARTIAL_SUPPORT = "PARTIAL_SUPPORT"
    TOPICAL_ONLY = "TOPICAL_ONLY"
    CONTRADICTS = "CONTRADICTS"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class MediatorFidelityVerdict(StrEnum):
    SUPPORTED = "SUPPORTED"
    PARTIAL = "PARTIAL"
    UNSUPPORTED = "UNSUPPORTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class MetricRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    value: Any
    n: int | None = Field(default=None, ge=0)
    unit: str | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    notes: str | None = None


class AxisResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    status: EvaluationStatus
    scope: str = Field(min_length=1)
    metrics: list[MetricRecord] = Field(default_factory=list)
    findings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)


class GroundingAuditItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    audit_id: str = Field(min_length=1)
    source_stage: Literal["Stage1", "Stage2", "Stage4", "Stage6"]
    case_id: str = Field(min_length=1)
    scenario_id: str | None = None
    lens: str | None = None
    claim_id: str | None = None
    claim_relation: str | None = None
    statement: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)
    evidence_records: list[dict[str, Any]] = Field(min_length=1)
    counter_evidence_records: list[dict[str, Any]] = Field(default_factory=list)
    analysis_cutoff_date: str | None = None
    deterministic_checks: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_evidence_alignment(self) -> "GroundingAuditItem":
        record_ids = {
            str(record.get("evidence_id"))
            for record in self.evidence_records
            if record.get("evidence_id")
        }
        if set(self.evidence_ids) - record_ids:
            raise ValueError("every evidence_id must have a supplied evidence record")
        return self


class GroundingAuditResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    audit_id: str = Field(min_length=1)
    verdict: GroundingVerdict
    direct_support: bool
    overgeneralization: bool
    unsupported_specificity: bool
    contradictory_evidence_omission: bool
    rationale: str = Field(min_length=1, max_length=1800)
    decisive_evidence_ids: list[str] = Field(default_factory=list)


class MediatorAuditItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    audit_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    round_index: int = Field(ge=1)
    adjudication_id: str = Field(min_length=1)
    outcome: str = Field(min_length=1)
    attack_claims: list[dict[str, Any]] = Field(default_factory=list)
    defense_claims: list[dict[str, Any]] = Field(default_factory=list)
    cited_evidence_records: list[dict[str, Any]] = Field(default_factory=list)
    rationale: str = Field(min_length=1)
    required_revision: str | None = None
    round_action: str = Field(min_length=1)
    next_round_focus: list[dict[str, Any]] = Field(default_factory=list)


class MediatorAuditResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    audit_id: str = Field(min_length=1)
    verdict: MediatorFidelityVerdict
    claim_evidence_fidelity: bool
    outcome_supported_by_supplied_evidence: bool
    required_revision_fidelity: bool
    round_action_fidelity: bool
    unseen_fact_detected: bool
    rationale: str = Field(min_length=1, max_length=1800)


class SingleAgentThreeLensResponse(BaseModel):
    """Single-agent architecture baseline: one model jointly emits all three Stage4 lenses."""

    model_config = ConfigDict(extra="forbid")

    decision_object_id: str = Field(min_length=1)
    technical_feasibility: LensAssessment
    institutional_regional: LensAssessment
    financial_adoption: LensAssessment

    @model_validator(mode="after")
    def validate_alignment(self) -> "SingleAgentThreeLensResponse":
        expected = {
            "technical_feasibility": LensType.TECHNICAL_FEASIBILITY,
            "institutional_regional": LensType.INSTITUTIONAL_REGIONAL,
            "financial_adoption": LensType.FINANCIAL_ADOPTION,
        }
        for field_name, lens_type in expected.items():
            assessment = getattr(self, field_name)
            if assessment.decision_object_id != self.decision_object_id:
                raise ValueError(f"{field_name} decision_object_id mismatch")
            if assessment.lens_type != lens_type:
                raise ValueError(f"{field_name} lens_type mismatch")
        return self


class VariantResultRecord(BaseModel):
    """Normalized result emitted by one of the final Stage 7 variant runs."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = STAGE7_VARIANT_RECORD_VERSION
    axis: str = Field(min_length=1)
    variant_id: str = Field(min_length=1)
    repeat_index: int | None = Field(default=None, ge=0)
    case_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    decision_object_id: str = Field(min_length=1)
    stance_by_lens: dict[str, int]
    d_lens: float = Field(ge=0.0, le=1.0)
    evidence_ids_by_lens: dict[str, list[str]] = Field(default_factory=dict)
    claim_directions_by_lens: dict[str, list[str]] = Field(default_factory=dict)
    recommendation: str = Field(min_length=1)
    prompt_hashes: dict[str, str] = Field(default_factory=dict)
    runtime: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_stances(self) -> "VariantResultRecord":
        if not self.stance_by_lens:
            raise ValueError("stance_by_lens must not be empty")
        if any(value not in (-1, 0, 1) for value in self.stance_by_lens.values()):
            raise ValueError("all lens stances must be -1, 0, or +1")
        return self


class Stage7ValidationArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(min_length=1)
    harness_version: str = Field(min_length=1)
    experiment_manifest_version: str = Field(min_length=1)
    created_at: str = Field(min_length=1)
    input_fingerprint: str = Field(min_length=1)
    main_case_count: int = Field(ge=0)
    main_scenario_count: int = Field(ge=0)
    source_stage6_artifact_sha256: dict[str, str]
    experiment_manifest_sha256: str = Field(min_length=1)
    evaluation_state_sha256: str = Field(min_length=1)
    runtime: dict[str, Any]
    axes: list[AxisResult]
    grounding_audits: list[dict[str, Any]] = Field(default_factory=list)
    mediator_audits: list[dict[str, Any]] = Field(default_factory=list)
    manual_audit_sample_path: str | None = None
    mediator_audit_sample_path: str | None = None
    artifact_sha256: str | None = None
