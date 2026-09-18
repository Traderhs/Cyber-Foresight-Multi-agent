from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from Stage4.schema import LensEvidenceSufficiency, LensType


STAGE5_DIAGNOSTIC_CONTRACT_VERSION = "stage5-diagnostics-v1"


class Stage5ValidationError(ValueError):
    """Raised when deterministic Stage 5 diagnostics violate their contract."""


class AgreementClass(StrEnum):
    GROUNDED_AGREEMENT = "GROUNDED_AGREEMENT"
    SUBSTANTIVE_DISAGREEMENT = "SUBSTANTIVE_DISAGREEMENT"
    EVIDENCE_LIMITED_DISAGREEMENT = "EVIDENCE_LIMITED_DISAGREEMENT"
    UNDER_INFORMED_CONVERGENCE = "UNDER_INFORMED_CONVERGENCE"


class ClaimEvidenceAuditStatus(StrEnum):
    DEFERRED_TO_STAGE7_SEMANTIC_GROUNDING_AUDIT = (
        "DEFERRED_TO_STAGE7_SEMANTIC_GROUNDING_AUDIT"
    )


class DirectionalClaimSignature(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["SUPPORTS_FEASIBILITY", "CHALLENGES_FEASIBILITY"]
    scope: Literal["GENERAL", "DEPLOYMENT_SPECIFIC"]
    evidence_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_ids(self) -> "DirectionalClaimSignature":
        if self.evidence_ids != sorted(set(self.evidence_ids)):
            raise ValueError("DirectionalClaimSignature evidence_ids must be sorted and unique")
        return self


class ScenarioLensDiagnostic(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision_object_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    jurisdiction_code: str = Field(min_length=1)
    stance_vector: dict[str, Literal[-1, 0, 1]]
    evidence_sufficiency_by_lens: dict[str, LensEvidenceSufficiency]
    d_lens: float = Field(ge=0.0, le=1.0)
    d_lens_evaluable: bool
    joint_abstention: bool
    nonzero_stance_lens_count: int = Field(ge=0, le=3)
    directional_lens_count: int = Field(ge=0, le=3)
    sufficient_or_partial_lens_count: int = Field(ge=0, le=3)
    agreement_class: AgreementClass
    support_claim_count_by_lens: dict[str, int]
    challenge_claim_count_by_lens: dict[str, int]
    cited_evidence_ids_by_lens: dict[str, list[str]]
    deployment_specific_directional_claim_count_by_lens: dict[str, int]

    @model_validator(mode="after")
    def validate_lens_keys(self) -> "ScenarioLensDiagnostic":
        expected = {lens.value for lens in LensType}
        fields = (
            "stance_vector",
            "evidence_sufficiency_by_lens",
            "support_claim_count_by_lens",
            "challenge_claim_count_by_lens",
            "cited_evidence_ids_by_lens",
            "deployment_specific_directional_claim_count_by_lens",
        )
        for field_name in fields:
            if set(getattr(self, field_name)) != expected:
                raise ValueError(f"{field_name} must contain exactly the three Stage 4 lenses")
        return self


class CritiqueDisagreementDiagnostic(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attack_pre_stance: Literal[-1, 0, 1]
    defense_pre_stance: Literal[-1, 0, 1]
    d_critique_pre: float = Field(ge=0.0, le=1.0)
    attack_post_stance: Literal[-1, 0, 1]
    defense_post_stance: Literal[-1, 0, 1]
    d_critique_post: float = Field(ge=0.0, le=1.0)
    d_critique_change: float = Field(ge=-1.0, le=1.0)
    unresolved_claim_count: int = Field(ge=0)
    evidence_gap_count: int = Field(ge=0)


class SystemEvidenceDiagnostic(BaseModel):
    model_config = ConfigDict(extra="forbid")

    required_evidence_slots: list[str]
    covered_evidence_slots: list[str]
    missing_evidence_slots: list[str]
    evidence_slot_coverage_ratio: float = Field(ge=0.0, le=1.0)
    source_family_count: int = Field(ge=0)
    independent_evidence_chain_count: int = Field(ge=0)
    temporal_violation_count: int = Field(ge=0)
    source_contract_violation_count: int = Field(ge=0)
    stage0_evidence_sufficiency: str = Field(min_length=1)
    stage4_sufficiency_counts: dict[str, int]
    stage4_claim_count: int = Field(ge=0)
    stage4_directional_claim_count: int = Field(ge=0)
    claim_evidence_reference_pass_rate: float = Field(ge=0.0, le=1.0)
    claim_evidence_audit_pass_rate: float | None = None
    claim_evidence_audit_status: ClaimEvidenceAuditStatus


class LensJurisdictionSensitivity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lens_type: LensType
    stance_by_jurisdiction: dict[str, Literal[-1, 0, 1]]
    stance_variance: float = Field(ge=0.0, le=1.0)
    jurisdiction_stance_sensitivity: bool
    available_lens_relevant_evidence_ids_by_jurisdiction: dict[str, list[str]]
    cited_directional_evidence_ids_by_jurisdiction: dict[str, list[str]]
    jurisdiction_specific_available_evidence_ids_by_jurisdiction: dict[str, list[str]]
    jurisdiction_specific_used_evidence_ids_by_jurisdiction: dict[str, list[str]]
    directional_claim_signatures_by_jurisdiction: dict[str, list[DirectionalClaimSignature]]
    deployment_specific_directional_claim_count_by_jurisdiction: dict[str, int]
    available_lens_relevant_evidence_changed: bool
    cited_directional_evidence_changed: bool
    directional_claim_signature_changed: bool
    deployment_specific_directional_claim_count_changed: bool
    jurisdiction_evidence_sensitivity: bool
    sensitivity_sources: list[str]

    @model_validator(mode="after")
    def validate_jurisdictions(self) -> "LensJurisdictionSensitivity":
        expected = {"KR", "EU", "US"}
        fields = (
            "stance_by_jurisdiction",
            "available_lens_relevant_evidence_ids_by_jurisdiction",
            "cited_directional_evidence_ids_by_jurisdiction",
            "jurisdiction_specific_available_evidence_ids_by_jurisdiction",
            "jurisdiction_specific_used_evidence_ids_by_jurisdiction",
            "directional_claim_signatures_by_jurisdiction",
            "deployment_specific_directional_claim_count_by_jurisdiction",
        )
        for field_name in fields:
            if set(getattr(self, field_name)) != expected:
                raise ValueError(f"{field_name} must contain exactly KR/EU/US")
        if self.sensitivity_sources != sorted(set(self.sensitivity_sources)):
            raise ValueError("sensitivity_sources must be sorted and unique")
        return self


class JurisdictionSensitivitySummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    by_lens: dict[str, LensJurisdictionSensitivity]
    any_stance_sensitivity: bool
    any_evidence_sensitivity: bool
    stance_sensitive_lenses: list[str]
    evidence_sensitive_lenses: list[str]

    @model_validator(mode="after")
    def validate_lens_map(self) -> "JurisdictionSensitivitySummary":
        expected = {lens.value for lens in LensType}
        if set(self.by_lens) != expected:
            raise ValueError("JurisdictionSensitivitySummary must contain exactly the three Stage 4 lenses")
        for values in (self.stance_sensitive_lenses, self.evidence_sensitive_lenses):
            if values != sorted(set(values)):
                raise ValueError("sensitivity lens lists must be sorted and unique")
        return self


class Stage5DiagnosticBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    diagnostic_contract_version: str = Field(min_length=1)
    scenario_diagnostics: list[ScenarioLensDiagnostic] = Field(min_length=3, max_length=3)
    critique_diagnostic: CritiqueDisagreementDiagnostic
    system_evidence_diagnostic: SystemEvidenceDiagnostic
    jurisdiction_sensitivity: JurisdictionSensitivitySummary
    model_reported_confidence_used: bool = False
    predictive_uncertainty_claimed_from_d_lens: bool = False

    @model_validator(mode="after")
    def validate_bundle(self) -> "Stage5DiagnosticBundle":
        if self.diagnostic_contract_version != STAGE5_DIAGNOSTIC_CONTRACT_VERSION:
            raise ValueError("Stage 5 diagnostic contract version mismatch")
        scenario_ids = [item.scenario_id for item in self.scenario_diagnostics]
        if set(scenario_ids) != {"KR__CIS_IG2", "EU__CIS_IG2", "US__CIS_IG2"}:
            raise ValueError("Stage 5 main bundle must contain exactly KR/EU/US CIS-IG2 diagnostics")
        if len(scenario_ids) != len(set(scenario_ids)):
            raise ValueError("Stage 5 scenario diagnostics must be unique")
        if self.model_reported_confidence_used:
            raise ValueError("Stage 5 must not use model-reported confidence as correctness probability")
        if self.predictive_uncertainty_claimed_from_d_lens:
            raise ValueError("D_lens must not be represented as Bayesian/predictive uncertainty")
        return self
