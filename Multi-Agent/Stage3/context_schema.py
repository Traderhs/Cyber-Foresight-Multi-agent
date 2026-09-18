from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from Stage0.context_scenarios import CONTEXT_SCENARIO_SET_VERSION, CONTEXT_SELECTION_RULE_VERSION
from Stage0.decision_evidence import (
    DECISION_EVIDENCE_RETRIEVAL_VERSION,
    DECISION_EVIDENCE_SLOT_CLAIM_TYPES,
)
from Stage3.schema import DecisionObject, Stage3ValidationError


STAGE3_CONTEXT_SEMANTIC_VALIDATION_VERSION = "stage3-context-semantic-validation-v2"


class ContextProvenanceType(StrEnum):
    SCENARIO_PARAMETER = "SCENARIO_PARAMETER"
    STANDARD_DERIVED = "STANDARD_DERIVED"
    OBSERVED_CONTEXT = "OBSERVED_CONTEXT"
    UNKNOWN = "UNKNOWN"


class ContextValue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str = Field(min_length=1)
    provenance_type: ContextProvenanceType
    source_evidence_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_sources(self) -> "ContextValue":
        if len(self.source_evidence_ids) != len(set(self.source_evidence_ids)):
            raise ValueError("ContextValue source_evidence_ids must be unique")
        if self.provenance_type == ContextProvenanceType.STANDARD_DERIVED and not self.source_evidence_ids:
            raise ValueError("STANDARD_DERIVED context requires source_evidence_ids")
        if self.provenance_type == ContextProvenanceType.SCENARIO_PARAMETER and self.source_evidence_ids:
            raise ValueError("SCENARIO_PARAMETER context must not masquerade as observed evidence")
        return self


class ContextScenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str = Field(min_length=1)
    scenario_set_version: str = Field(min_length=1)
    context_selection_rule: str = Field(min_length=1)
    jurisdiction_code: str = Field(min_length=1)
    jurisdiction_source_registry_id: str = Field(min_length=1)
    region: ContextValue
    organization_type: ContextValue
    infrastructure_context: ContextValue
    budget_context: ContextValue
    reference_enterprise_profile: ContextValue
    assumptions: list[str] = Field(default_factory=list)
    non_assumptions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_lists(self) -> "ContextScenario":
        if self.scenario_set_version != CONTEXT_SCENARIO_SET_VERSION:
            raise ValueError("ContextScenario scenario_set_version mismatch")
        if self.context_selection_rule != CONTEXT_SELECTION_RULE_VERSION:
            raise ValueError("ContextScenario context_selection_rule mismatch")
        for name, values in (("assumptions", self.assumptions), ("non_assumptions", self.non_assumptions)):
            if any(not item.strip() for item in values):
                raise ValueError(f"{name} cannot contain blank values")
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must contain unique values")
        return self


class DecisionEvidencePack(BaseModel):
    model_config = ConfigDict(extra="forbid")

    retrieval_version: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    source_snapshot_id: str = Field(min_length=1)
    decision_source_snapshot_id: str | None = None
    decision_source_manifest_sha256: str | None = None
    decision_source_records_sha256: str | None = None
    action_evidence_snapshot_id: str | None = None
    action_evidence_registry_version: str | None = None
    action_evidence_registry_sha256: str | None = None
    main_guidance_selection_version: str | None = None
    main_guidance_selection_sha256: str | None = None
    analysis_cutoff_date: str = Field(min_length=1)
    context_evidence_records: list[dict[str, Any]] = Field(min_length=1)
    decision_evidence_records: list[dict[str, Any]] = Field(default_factory=list)
    context_evidence_ids: list[str] = Field(min_length=1)
    decision_evidence_ids: list[str] = Field(default_factory=list)
    slot_coverage: dict[str, list[str]] = Field(default_factory=dict)
    retrieval_metadata: dict[str, Any]

    @model_validator(mode="after")
    def validate_records(self) -> "DecisionEvidencePack":
        if self.retrieval_version != DECISION_EVIDENCE_RETRIEVAL_VERSION:
            raise ValueError("DecisionEvidencePack retrieval_version mismatch")
        try:
            cutoff = date.fromisoformat(self.analysis_cutoff_date)
        except ValueError as exc:
            raise ValueError("DecisionEvidencePack analysis_cutoff_date must be ISO YYYY-MM-DD") from exc
        records = [*self.context_evidence_records, *self.decision_evidence_records]
        ids = [str(record.get("evidence_id") or "") for record in records]
        if any(not item for item in ids):
            raise ValueError("DecisionEvidencePack records require evidence_id")
        if len(ids) != len(set(ids)):
            raise ValueError("DecisionEvidencePack evidence IDs must be unique")
        if set(self.context_evidence_ids) != {
            str(record["evidence_id"]) for record in self.context_evidence_records
        }:
            raise ValueError("context_evidence_ids must exactly match context_evidence_records")
        if set(self.decision_evidence_ids) != {
            str(record["evidence_id"]) for record in self.decision_evidence_records
        }:
            raise ValueError("decision_evidence_ids must exactly match decision_evidence_records")
        if set(self.slot_coverage) - set(self.decision_evidence_ids):
            raise ValueError("slot_coverage cannot reference evidence outside decision_evidence_ids")
        for record in records:
            available_at = str(record.get("available_at") or "")
            if not available_at:
                raise ValueError("DecisionEvidencePack records require available_at")
            try:
                available_date = date.fromisoformat(available_at)
            except ValueError as exc:
                raise ValueError("DecisionEvidencePack record available_at must be ISO YYYY-MM-DD") from exc
            if available_date > cutoff:
                raise ValueError(
                    f"DecisionEvidencePack temporal violation: {record.get('evidence_id')} available after cutoff"
                )
        allowed_decision_snapshots = {self.source_snapshot_id}
        if self.decision_source_snapshot_id:
            allowed_decision_snapshots.add(self.decision_source_snapshot_id)
            if not self.decision_source_manifest_sha256 or not self.decision_source_records_sha256:
                raise ValueError("DecisionEvidencePack decision-source snapshot requires manifest and records hashes")
        elif self.decision_source_manifest_sha256 or self.decision_source_records_sha256:
            raise ValueError("DecisionEvidencePack decision-source hashes require snapshot ID")
        if self.action_evidence_snapshot_id:
            allowed_decision_snapshots.add(self.action_evidence_snapshot_id)
            if not self.action_evidence_registry_version or not self.action_evidence_registry_sha256:
                raise ValueError("DecisionEvidencePack action-evidence snapshot requires registry version/hash")
        elif self.action_evidence_registry_version or self.action_evidence_registry_sha256:
            raise ValueError("DecisionEvidencePack action-evidence registry metadata requires snapshot ID")
        if bool(self.main_guidance_selection_version) != bool(self.main_guidance_selection_sha256):
            raise ValueError(
                "DecisionEvidencePack main-guidance selection version/hash must be provided together"
            )
        for record in self.decision_evidence_records:
            if str(record.get("source_snapshot_id") or "") not in allowed_decision_snapshots:
                raise ValueError("DecisionEvidencePack decision record source_snapshot_id mismatch")
        for record in self.context_evidence_records:
            if str(record.get("source_snapshot_id") or "") != CONTEXT_SCENARIO_SET_VERSION:
                raise ValueError("DecisionEvidencePack context record scenario-source version mismatch")
        valid_slots = set(DECISION_EVIDENCE_SLOT_CLAIM_TYPES)
        referenced_slots = {slot for slots in self.slot_coverage.values() for slot in slots}
        if referenced_slots - valid_slots:
            raise ValueError("slot_coverage contains an unknown decision-evidence slot")
        metadata_required = set(self.retrieval_metadata.get("required_slots") or [])
        metadata_covered = set(self.retrieval_metadata.get("covered_slots") or [])
        metadata_missing = set(self.retrieval_metadata.get("missing_slots") or [])
        if metadata_required != valid_slots:
            raise ValueError("retrieval_metadata required_slots mismatch")
        if metadata_covered != referenced_slots:
            raise ValueError("retrieval_metadata covered_slots mismatch")
        if metadata_missing != valid_slots - referenced_slots:
            raise ValueError("retrieval_metadata missing_slots mismatch")
        return self


class ContextualDecisionObject(DecisionObject):
    model_config = ConfigDict(extra="forbid")

    base_decision_object_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    scenario_set_version: str = Field(min_length=1)
    context_scenario: ContextScenario
    forecast_evidence_ids: list[str] = Field(min_length=1)
    context_evidence_ids: list[str] = Field(min_length=1)
    decision_evidence_ids: list[str] = Field(default_factory=list)
    evaluation_evidence_records: list[dict[str, Any]] = Field(min_length=1)
    decision_evidence_pack: DecisionEvidencePack

    @model_validator(mode="after")
    def validate_context_contract(self) -> "ContextualDecisionObject":
        if self.scenario_id != self.context_scenario.scenario_id:
            raise ValueError("scenario_id must match context_scenario.scenario_id")
        if self.scenario_set_version != self.context_scenario.scenario_set_version:
            raise ValueError("scenario_set_version mismatch")
        if self.scenario_id != self.decision_evidence_pack.scenario_id:
            raise ValueError("DecisionEvidencePack scenario_id mismatch")
        if self.deployment_context.region != self.context_scenario.region.value:
            raise ValueError("deployment_context.region mismatch")
        if self.deployment_context.organization_type != self.context_scenario.organization_type.value:
            raise ValueError("deployment_context.organization_type mismatch")
        if self.deployment_context.infrastructure_context != self.context_scenario.infrastructure_context.value:
            raise ValueError("deployment_context.infrastructure_context mismatch")
        if self.deployment_context.budget_context != self.context_scenario.budget_context.value:
            raise ValueError("deployment_context.budget_context mismatch")
        if self.unknown_context_fields:
            raise ValueError("Main contextual DecisionObject cannot retain unknown deployment fields")
        for name, values in (
            ("forecast_evidence_ids", self.forecast_evidence_ids),
            ("context_evidence_ids", self.context_evidence_ids),
            ("decision_evidence_ids", self.decision_evidence_ids),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must contain unique values")
        record_ids = [str(record.get("evidence_id") or "") for record in self.evaluation_evidence_records]
        if any(not item for item in record_ids) or len(record_ids) != len(set(record_ids)):
            raise ValueError("evaluation_evidence_records must contain unique non-empty evidence IDs")
        if set(record_ids) != set(self.evaluation_evidence_ids):
            raise ValueError("evaluation_evidence_ids must exactly match evaluation_evidence_records")
        expected_union = set(self.forecast_evidence_ids) | set(self.context_evidence_ids) | set(self.decision_evidence_ids)
        if expected_union != set(self.evaluation_evidence_ids):
            raise ValueError("evaluation_evidence_ids must equal forecast/context/decision evidence union")
        if set(self.context_evidence_ids) != set(self.decision_evidence_pack.context_evidence_ids):
            raise ValueError("context evidence mismatch")
        if set(self.decision_evidence_ids) != set(self.decision_evidence_pack.decision_evidence_ids):
            raise ValueError("decision evidence mismatch")
        return self


class Stage3ContextualDecisionBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    base_stage3_input_fingerprint: str = Field(min_length=1)
    scenario_set_version: str = Field(min_length=1)
    contextual_decision_objects: list[ContextualDecisionObject] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_bundle(self) -> "Stage3ContextualDecisionBundle":
        if self.scenario_set_version != CONTEXT_SCENARIO_SET_VERSION:
            raise ValueError("Stage3 contextual scenario-set version mismatch")
        if any(item.case_id != self.case_id for item in self.contextual_decision_objects):
            raise ValueError("ContextualDecisionObject case_id mismatch")
        scenario_ids = [item.scenario_id for item in self.contextual_decision_objects]
        if len(scenario_ids) != len(set(scenario_ids)):
            raise ValueError("Contextual scenario IDs must be unique")
        if any(item.scenario_set_version != self.scenario_set_version for item in self.contextual_decision_objects):
            raise ValueError("Contextual scenario-set version mismatch")
        if set(scenario_ids) != {"KR__CIS_IG2", "EU__CIS_IG2", "US__CIS_IG2"}:
            raise ValueError("Main contextual bundle must contain exactly KR/EU/US CIS-IG2 scenarios")
        return self


def validate_contextual_bundle(value: dict[str, Any]) -> Stage3ContextualDecisionBundle:
    try:
        return Stage3ContextualDecisionBundle.model_validate(value)
    except Exception as exc:
        raise Stage3ValidationError(f"Invalid contextual Stage 3 bundle: {exc}") from exc

