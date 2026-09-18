from __future__ import annotations

import copy
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from Stage3.schema import DecisionObject


STAGE4_SEMANTIC_VALIDATION_VERSION = "stage4-semantic-validation-v1"


class Stage4ValidationError(ValueError):
    """Raised when a Stage 4 lens output violates the frozen evaluation contract."""


class LensType(StrEnum):
    TECHNICAL_FEASIBILITY = "technical_feasibility"
    INSTITUTIONAL_REGIONAL = "institutional_regional"
    FINANCIAL_ADOPTION = "financial_adoption"


class LensEvidenceSufficiency(StrEnum):
    SUFFICIENT = "SUFFICIENT"
    PARTIAL = "PARTIAL"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    INSUFFICIENT_CONTEXT = "INSUFFICIENT_CONTEXT"


class LensClaimStatus(StrEnum):
    SUPPORTS_FEASIBILITY = "SUPPORTS_FEASIBILITY"
    CHALLENGES_FEASIBILITY = "CHALLENGES_FEASIBILITY"
    NEUTRAL_CONTEXT = "NEUTRAL_CONTEXT"


class LensClaimScope(StrEnum):
    GENERAL = "GENERAL"
    DEPLOYMENT_SPECIFIC = "DEPLOYMENT_SPECIFIC"


class LensClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)
    status: LensClaimStatus
    scope: LensClaimScope

    @model_validator(mode="after")
    def validate_evidence_ids(self) -> "LensClaim":
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("LensClaim evidence_ids must be unique")
        return self


class LensAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision_object_id: str = Field(min_length=1)
    lens_type: LensType
    stance: Literal[-1, 0, 1]
    evidence_sufficiency: LensEvidenceSufficiency
    rationale: str = Field(min_length=1)
    claims: list[LensClaim] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    evidence_gaps: list[str] = Field(default_factory=list)
    conditional_requirements: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_internal_contract(self) -> "LensAssessment":
        claim_ids = [claim.claim_id for claim in self.claims]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("LensAssessment claim_id values must be unique")
        for name, values in (
            ("constraints", self.constraints),
            ("evidence_gaps", self.evidence_gaps),
            ("conditional_requirements", self.conditional_requirements),
        ):
            if any(not str(value).strip() for value in values):
                raise ValueError(f"{name} cannot contain blank values")
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must contain unique values")

        support_claims = [
            claim for claim in self.claims if claim.status == LensClaimStatus.SUPPORTS_FEASIBILITY
        ]
        challenge_claims = [
            claim for claim in self.claims if claim.status == LensClaimStatus.CHALLENGES_FEASIBILITY
        ]
        if self.stance == 1 and not support_claims:
            raise ValueError("stance=+1 requires at least one SUPPORTS_FEASIBILITY claim")
        if self.stance == -1 and not challenge_claims:
            raise ValueError("stance=-1 requires at least one CHALLENGES_FEASIBILITY claim")
        if self.evidence_sufficiency in {
            LensEvidenceSufficiency.INSUFFICIENT_EVIDENCE,
            LensEvidenceSufficiency.INSUFFICIENT_CONTEXT,
        } and self.stance != 0:
            raise ValueError("Insufficient evidence/context requires stance=0")
        if (
            self.evidence_sufficiency
            in {LensEvidenceSufficiency.SUFFICIENT, LensEvidenceSufficiency.PARTIAL}
            and not self.claims
        ):
            raise ValueError("SUFFICIENT/PARTIAL assessment requires at least one evidence-backed claim")
        if self.evidence_sufficiency == LensEvidenceSufficiency.INSUFFICIENT_EVIDENCE and not self.evidence_gaps:
            raise ValueError("INSUFFICIENT_EVIDENCE requires an explicit evidence_gaps entry")
        if self.evidence_sufficiency == LensEvidenceSufficiency.INSUFFICIENT_CONTEXT and not (
            self.evidence_gaps or self.conditional_requirements
        ):
            raise ValueError(
                "INSUFFICIENT_CONTEXT requires an explicit gap or conditional requirement"
            )
        return self


class DecisionLensEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision_object_id: str = Field(min_length=1)
    technical_feasibility: LensAssessment
    institutional_regional: LensAssessment
    financial_adoption: LensAssessment

    @model_validator(mode="after")
    def validate_alignment(self) -> "DecisionLensEvaluation":
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


class Stage4EvaluationBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    evaluations: list[DecisionLensEvaluation] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_objects(self) -> "Stage4EvaluationBundle":
        object_ids = [item.decision_object_id for item in self.evaluations]
        if len(object_ids) != len(set(object_ids)):
            raise ValueError("Stage4EvaluationBundle decision_object_id values must be unique")
        return self


def _evidence_ids(evidence_pack: dict[str, Any]) -> list[str]:
    return sorted(
        {
            str(record.get("evidence_id"))
            for record in evidence_pack.get("evidence", [])
            if record.get("evidence_id")
        }
    )


def build_constrained_lens_response_schema(
    *,
    decision_object: DecisionObject | dict[str, Any],
    evidence_pack: dict[str, Any],
    expected_lens_type: LensType,
) -> dict[str, Any]:
    obj = (
        decision_object
        if isinstance(decision_object, DecisionObject)
        else DecisionObject.model_validate(decision_object)
    )
    allowed_ids = sorted(set(obj.evaluation_evidence_ids))
    if not allowed_ids:
        raise Stage4ValidationError("Stage 4 requires a non-empty frozen evaluation evidence set")
    pack_ids = set(_evidence_ids(evidence_pack))
    outside = sorted(set(allowed_ids) - pack_ids)
    if outside:
        raise Stage4ValidationError(
            f"DecisionObject evaluation_evidence_ids are missing from Stage 0 EvidencePack: {outside}"
        )

    schema = copy.deepcopy(LensAssessment.model_json_schema())
    schema["properties"]["decision_object_id"] = {
        "type": "string",
        "const": obj.decision_object_id,
    }
    schema["properties"]["lens_type"] = {
        "type": "string",
        "const": expected_lens_type.value,
    }
    schema["$defs"]["LensClaim"]["properties"]["evidence_ids"]["items"] = {
        "type": "string",
        "enum": allowed_ids,
    }
    return schema


def validate_lens_assessment(
    assessment: LensAssessment | dict[str, Any],
    *,
    decision_object: DecisionObject | dict[str, Any],
    evidence_pack: dict[str, Any],
    expected_lens_type: LensType,
) -> LensAssessment:
    try:
        parsed = (
            assessment
            if isinstance(assessment, LensAssessment)
            else LensAssessment.model_validate(assessment)
        )
        obj = (
            decision_object
            if isinstance(decision_object, DecisionObject)
            else DecisionObject.model_validate(decision_object)
        )
    except Exception as exc:
        raise Stage4ValidationError(f"Invalid Stage 4 lens schema: {exc}") from exc

    errors: list[str] = []
    if parsed.decision_object_id != obj.decision_object_id:
        errors.append(
            f"decision_object_id mismatch: expected={obj.decision_object_id!r}, "
            f"actual={parsed.decision_object_id!r}"
        )
    if parsed.lens_type != expected_lens_type:
        errors.append(
            f"lens_type mismatch: expected={expected_lens_type.value!r}, "
            f"actual={parsed.lens_type.value!r}"
        )

    pack_ids = set(_evidence_ids(evidence_pack))
    allowed_ids = set(obj.evaluation_evidence_ids)
    if allowed_ids != pack_ids:
        errors.append(
            "Stage 4 main contract requires DecisionObject evaluation_evidence_ids to equal the frozen "
            "Stage 0 EvidencePack evidence set"
        )
    referenced = {
        evidence_id
        for claim in parsed.claims
        for evidence_id in claim.evidence_ids
    }
    outside = sorted(referenced - allowed_ids)
    if outside:
        errors.append(f"LensAssessment references evidence outside the frozen evaluation set: {outside}")

    context = obj.deployment_context
    missing_relevant_context = {
        LensType.TECHNICAL_FEASIBILITY: context.infrastructure_context in (None, ""),
        LensType.INSTITUTIONAL_REGIONAL: context.region in (None, ""),
        LensType.FINANCIAL_ADOPTION: (
            context.organization_type in (None, "")
            or context.infrastructure_context in (None, "")
            or context.budget_context in (None, "")
        ),
    }[expected_lens_type]
    deployment_specific = [
        claim.claim_id
        for claim in parsed.claims
        if claim.scope == LensClaimScope.DEPLOYMENT_SPECIFIC
    ]
    if missing_relevant_context and deployment_specific:
        errors.append(
            f"{expected_lens_type.value} cannot emit DEPLOYMENT_SPECIFIC claims when its required deployment "
            f"context is missing; invalid claim IDs: {deployment_specific}"
        )

    if errors:
        raise Stage4ValidationError("; ".join(errors))
    return parsed


def validate_stage4_evaluation_bundle(
    bundle: Stage4EvaluationBundle | dict[str, Any],
    *,
    decision_objects: list[DecisionObject | dict[str, Any]],
    evidence_pack: dict[str, Any],
) -> Stage4EvaluationBundle:
    try:
        parsed = (
            bundle
            if isinstance(bundle, Stage4EvaluationBundle)
            else Stage4EvaluationBundle.model_validate(bundle)
        )
        objects = [
            item if isinstance(item, DecisionObject) else DecisionObject.model_validate(item)
            for item in decision_objects
        ]
    except Exception as exc:
        raise Stage4ValidationError(f"Invalid Stage4EvaluationBundle schema: {exc}") from exc

    if not objects:
        raise Stage4ValidationError("Stage 4 requires at least one DecisionObject")
    expected_case_id = objects[0].case_id
    if any(item.case_id != expected_case_id for item in objects):
        raise Stage4ValidationError("Stage 4 DecisionObjects must belong to one case")
    if parsed.case_id != expected_case_id:
        raise Stage4ValidationError("Stage4EvaluationBundle case_id mismatch")

    expected_ids = [item.decision_object_id for item in objects]
    actual_ids = [item.decision_object_id for item in parsed.evaluations]
    if sorted(expected_ids) != sorted(actual_ids):
        raise Stage4ValidationError(
            "Stage 4 must emit exactly one three-lens evaluation per frozen DecisionObject"
        )
    objects_by_id = {item.decision_object_id: item for item in objects}
    for evaluation in parsed.evaluations:
        obj = objects_by_id[evaluation.decision_object_id]
        validate_lens_assessment(
            evaluation.technical_feasibility,
            decision_object=obj,
            evidence_pack=evidence_pack,
            expected_lens_type=LensType.TECHNICAL_FEASIBILITY,
        )
        validate_lens_assessment(
            evaluation.institutional_regional,
            decision_object=obj,
            evidence_pack=evidence_pack,
            expected_lens_type=LensType.INSTITUTIONAL_REGIONAL,
        )
        validate_lens_assessment(
            evaluation.financial_adoption,
            decision_object=obj,
            evidence_pack=evidence_pack,
            expected_lens_type=LensType.FINANCIAL_ADOPTION,
        )
    return parsed

