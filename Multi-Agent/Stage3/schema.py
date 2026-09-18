from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from Stage2.schema import EvidenceGap, MediatorAdjudication


STAGE3_SEMANTIC_VALIDATION_VERSION = "stage3-semantic-validation-v2"
STAGE3_SELECTION_RULE_VERSION = "case-forecast-pmt-only-v1"


class Stage3ValidationError(ValueError):
    """Raised when a Stage 3 deterministic object violates the fixed contract."""


class CandidateActionSource(StrEnum):
    B_MTGNN_CASE_PMT = "B_MTGNN_CASE_PMT"


class CandidateActionRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_id: str = Field(min_length=1)
    action_name: str = Field(min_length=1)
    source: CandidateActionSource
    source_evidence_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_evidence(self) -> "CandidateActionRef":
        if len(self.source_evidence_ids) != len(set(self.source_evidence_ids)):
            raise ValueError("CandidateActionRef source_evidence_ids must be unique")
        return self


class ExcludedAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class TimingWindow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_year: int
    end_year: int

    @model_validator(mode="after")
    def validate_order(self) -> "TimingWindow":
        if self.start_year > self.end_year:
            raise ValueError("TimingWindow start_year must be <= end_year")
        return self


class DeploymentContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    region: str | None = None
    organization_type: str | None = None
    infrastructure_context: str | None = None
    budget_context: str | None = None


class RequiredRevisionRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    adjudication_id: str = Field(min_length=1)
    outcome: str = Field(min_length=1)
    attack_claim_ids: list[str] = Field(default_factory=list)
    defense_claim_ids: list[str] = Field(default_factory=list)
    required_revision: str = Field(min_length=1)


class Stage2CritiqueSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attack_post_stance: int = Field(ge=-1, le=1)
    defense_post_stance: int = Field(ge=-1, le=1)
    final_mediator_adjudications: list[MediatorAdjudication] = Field(min_length=1)
    matched_resolved_claim_ids: list[str] = Field(default_factory=list)
    matched_unresolved_claim_ids: list[str] = Field(default_factory=list)
    required_revisions: list[RequiredRevisionRef] = Field(default_factory=list)
    attack_post_unresolved_questions: list[str] = Field(default_factory=list)
    defense_post_unresolved_questions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_claim_ids(self) -> "Stage2CritiqueSummary":
        for name, values in (
            ("matched_resolved_claim_ids", self.matched_resolved_claim_ids),
            ("matched_unresolved_claim_ids", self.matched_unresolved_claim_ids),
            ("attack_post_unresolved_questions", self.attack_post_unresolved_questions),
            ("defense_post_unresolved_questions", self.defense_post_unresolved_questions),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must contain unique values")
        overlap = set(self.matched_resolved_claim_ids).intersection(self.matched_unresolved_claim_ids)
        if overlap:
            raise ValueError(f"Matched claim IDs cannot be both resolved and unresolved: {sorted(overlap)}")
        return self


class DecisionObject(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision_object_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    threat: str = Field(min_length=1)
    threat_id: str = Field(min_length=1)
    pmt_or_mitigation: str = Field(min_length=1)
    pmt_or_mitigation_id: str = Field(min_length=1)
    forecast_interpretation: dict[str, Any]
    stage2_critique_summary: Stage2CritiqueSummary
    candidate_action: CandidateActionRef
    candidate_action_source: CandidateActionSource
    candidate_selection_rule: str = Field(min_length=1)
    intended_goal: str = Field(min_length=1)
    timing_window: TimingWindow
    deployment_context: DeploymentContext
    known_constraints: list[str] = Field(default_factory=list)
    unknown_context_fields: list[str] = Field(default_factory=list)
    evaluation_evidence_ids: list[str] = Field(min_length=1)
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    contradictory_evidence_ids: list[str] = Field(default_factory=list)
    unresolved_evidence_gaps: list[EvidenceGap] = Field(default_factory=list)
    predictive_uncertainty: dict[str, Any] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_internal_contract(self) -> "DecisionObject":
        if self.candidate_action.action_id != self.pmt_or_mitigation_id:
            raise ValueError("Stage 3 v1 candidate_action must be the case forecast PMT")
        if self.candidate_action.action_name != self.pmt_or_mitigation:
            raise ValueError("candidate_action name must match pmt_or_mitigation")
        if self.candidate_action.source != self.candidate_action_source:
            raise ValueError("candidate_action_source must match candidate_action.source")
        if self.candidate_selection_rule != STAGE3_SELECTION_RULE_VERSION:
            raise ValueError("DecisionObject candidate_selection_rule mismatch")
        for name, values in (
            ("known_constraints", self.known_constraints),
            ("unknown_context_fields", self.unknown_context_fields),
            ("evaluation_evidence_ids", self.evaluation_evidence_ids),
            ("supporting_evidence_ids", self.supporting_evidence_ids),
            ("contradictory_evidence_ids", self.contradictory_evidence_ids),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must contain unique values")
        expected_unknown = {
            field
            for field in ("region", "organization_type", "infrastructure_context", "budget_context")
            if getattr(self.deployment_context, field) in (None, "")
        }
        if set(self.unknown_context_fields) != expected_unknown:
            raise ValueError(
                "unknown_context_fields must exactly match missing deployment_context fields"
            )
        evaluation_ids = set(self.evaluation_evidence_ids)
        outside_support = sorted(set(self.supporting_evidence_ids) - evaluation_ids)
        outside_contradiction = sorted(set(self.contradictory_evidence_ids) - evaluation_ids)
        if outside_support or outside_contradiction:
            raise ValueError(
                "supporting/contradictory evidence must be contained in evaluation_evidence_ids: "
                f"support={outside_support}, contradiction={outside_contradiction}"
            )
        return self


class Stage3DecisionBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    candidate_selection_rule: str = Field(min_length=1)
    eligible_action_set: list[CandidateActionRef] = Field(min_length=1)
    excluded_actions_and_reason: list[ExcludedAction] = Field(default_factory=list)
    decision_objects: list[DecisionObject] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_bundle(self) -> "Stage3DecisionBundle":
        if self.candidate_selection_rule != STAGE3_SELECTION_RULE_VERSION:
            raise ValueError("Stage3DecisionBundle selection rule mismatch")
        if self.candidate_selection_rule == "case-forecast-pmt-only-v1" and len(self.eligible_action_set) != 1:
            raise ValueError("case-forecast-pmt-only-v1 requires exactly one eligible action")
        action_ids = [item.action_id for item in self.eligible_action_set]
        if len(action_ids) != len(set(action_ids)):
            raise ValueError("eligible_action_set action IDs must be unique")
        excluded_ids = [item.action_id for item in self.excluded_actions_and_reason]
        if len(excluded_ids) != len(set(excluded_ids)):
            raise ValueError("excluded_actions_and_reason action IDs must be unique")
        overlap = sorted(set(action_ids).intersection(excluded_ids))
        if overlap:
            raise ValueError(f"Actions cannot be both eligible and excluded: {overlap}")
        object_action_ids = [item.candidate_action.action_id for item in self.decision_objects]
        if sorted(object_action_ids) != sorted(action_ids):
            raise ValueError("Stage 3 must emit exactly one DecisionObject per eligible action")
        if any(item.case_id != self.case_id for item in self.decision_objects):
            raise ValueError("DecisionObject case_id mismatch within Stage3DecisionBundle")
        return self
