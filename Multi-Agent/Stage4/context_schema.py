from __future__ import annotations

import copy
from typing import Any

from pydantic import model_validator

from Stage3.context_schema import ContextualDecisionObject
from Stage4.schema import (
    DecisionLensEvaluation,
    LensAssessment,
    LensClaimStatus,
    LensClaimScope,
    LensEvidenceSufficiency,
    LensType,
    Stage4EvaluationBundle,
    Stage4ValidationError,
)


STAGE4_CONTEXT_SEMANTIC_VALIDATION_VERSION = "stage4-semantic-validation-v6"


# Contextual Stage 4 v4 makes the grounding contract scope- and direction-aware. A GENERAL
# institutional claim may legitimately be directional when action-specific
# governance/audit burden is documented, while a DEPLOYMENT_SPECIFIC regional
# claim still requires jurisdiction-applicability evidence.  The other two
# lenses retain the same action-specific slot families at both scopes.
_DIRECTIONAL_SLOTS_BY_LENS_AND_SCOPE: dict[
    LensType, dict[LensClaimScope, set[str]]
] = {
    LensType.TECHNICAL_FEASIBILITY: {
        LensClaimScope.GENERAL: {
            "technical_enablement",
            "technical_limitation",
            "deployment_maturity",
        },
        LensClaimScope.DEPLOYMENT_SPECIFIC: {
            "technical_enablement",
            "technical_limitation",
            "deployment_maturity",
        },
    },
    LensType.INSTITUTIONAL_REGIONAL: {
        LensClaimScope.GENERAL: {
            "governance_enablement",
            "compliance_constraint",
            "regulatory_applicability",
        },
        LensClaimScope.DEPLOYMENT_SPECIFIC: {
            "governance_enablement",
            "compliance_constraint",
            "regulatory_applicability",
        },
    },
    LensType.FINANCIAL_ADOPTION: {
        LensClaimScope.GENERAL: {
            "adoption_benefit",
            "adoption_burden",
            "deployment_maturity",
        },
        LensClaimScope.DEPLOYMENT_SPECIFIC: {
            "adoption_benefit",
            "adoption_burden",
            "deployment_maturity",
        },
    },
}


def contextual_general_evidence_signature(
    decision_object: ContextualDecisionObject,
    lens_type: LensType,
) -> tuple[str, ...]:
    """Return the exact GENERAL lens-relevant evidence signature.

    Jurisdiction labels are scenario parameters, not evidence. Two scenario
    objects with the same signature expose the same evidence capable of
    carrying (or materially qualifying) GENERAL direction for this lens.
    """

    relevant_slots = _DIRECTIONAL_SLOTS_BY_LENS_AND_SCOPE[lens_type][LensClaimScope.GENERAL]
    return tuple(
        sorted(
            evidence_id
            for evidence_id, slots in decision_object.decision_evidence_pack.slot_coverage.items()
            if set(slots) & relevant_slots
        )
    )


_DIRECTIONAL_SLOTS_BY_LENS_STATUS: dict[
    LensType, dict[LensClaimStatus, set[str]]
] = {
    LensType.TECHNICAL_FEASIBILITY: {
        LensClaimStatus.SUPPORTS_FEASIBILITY: {"technical_enablement"},
        LensClaimStatus.CHALLENGES_FEASIBILITY: {"technical_limitation"},
    },
    LensType.INSTITUTIONAL_REGIONAL: {
        LensClaimStatus.SUPPORTS_FEASIBILITY: {"governance_enablement"},
        LensClaimStatus.CHALLENGES_FEASIBILITY: {"compliance_constraint"},
    },
    LensType.FINANCIAL_ADOPTION: {
        LensClaimStatus.SUPPORTS_FEASIBILITY: {"adoption_benefit"},
        LensClaimStatus.CHALLENGES_FEASIBILITY: {"adoption_burden"},
    },
}


_REQUIRED_SCOPE_SLOTS_BY_LENS: dict[
    LensType, dict[LensClaimScope, set[str]]
] = {
    LensType.TECHNICAL_FEASIBILITY: {
        LensClaimScope.GENERAL: set(),
        LensClaimScope.DEPLOYMENT_SPECIFIC: set(),
    },
    LensType.INSTITUTIONAL_REGIONAL: {
        LensClaimScope.GENERAL: set(),
        LensClaimScope.DEPLOYMENT_SPECIFIC: {"regulatory_applicability"},
    },
    LensType.FINANCIAL_ADOPTION: {
        LensClaimScope.GENERAL: set(),
        LensClaimScope.DEPLOYMENT_SPECIFIC: set(),
    },
}


def contextual_directional_slots(
    lens_type: LensType,
    scope: LensClaimScope,
    status: LensClaimStatus | None = None,
) -> set[str]:
    """Return the frozen v4 slots allowed to carry this lens/scope/direction."""

    slots = set(_DIRECTIONAL_SLOTS_BY_LENS_AND_SCOPE[lens_type][scope])
    if status in {
        LensClaimStatus.SUPPORTS_FEASIBILITY,
        LensClaimStatus.CHALLENGES_FEASIBILITY,
    }:
        slots &= _DIRECTIONAL_SLOTS_BY_LENS_STATUS[lens_type][status]
    return slots


def contextual_required_scope_slots(
    lens_type: LensType,
    scope: LensClaimScope,
) -> set[str]:
    """Return extra scope evidence that must accompany a direction carrier."""

    return set(_REQUIRED_SCOPE_SLOTS_BY_LENS[lens_type][scope])


def context_lens_equivalence_key(
    decision_object: ContextualDecisionObject | dict[str, Any],
    lens_type: LensType,
) -> tuple[Any, ...]:
    """Return the deterministic cross-jurisdiction equivalence key for one lens.

    A jurisdiction label is not factual evidence. Technical and Financial
    assessments are therefore jurisdiction-neutral whenever their lens-relevant
    evidence and non-jurisdiction deployment context are identical.
    Institutional assessments are jurisdiction-neutral only when no
    regulatory_applicability evidence is present; once such evidence exists,
    the jurisdiction code becomes part of the key and those scenarios must be
    evaluated separately.
    """

    obj = (
        decision_object
        if isinstance(decision_object, ContextualDecisionObject)
        else ContextualDecisionObject.model_validate(decision_object)
    )
    relevant_slots = _DIRECTIONAL_SLOTS_BY_LENS_AND_SCOPE[lens_type][LensClaimScope.GENERAL]
    evidence_ids = tuple(
        sorted(
            evidence_id
            for evidence_id, slots in obj.decision_evidence_pack.slot_coverage.items()
            if set(slots) & relevant_slots
        )
    )
    regulatory_ids = tuple(
        sorted(
            evidence_id
            for evidence_id, slots in obj.decision_evidence_pack.slot_coverage.items()
            if "regulatory_applicability" in set(slots)
        )
    )
    non_jurisdiction_context = (
        obj.deployment_context.organization_type,
        obj.deployment_context.infrastructure_context,
        obj.deployment_context.budget_context,
        obj.context_scenario.reference_enterprise_profile.value,
    )
    jurisdiction_bound = (
        lens_type == LensType.INSTITUTIONAL_REGIONAL and bool(regulatory_ids)
    )
    return (
        "JURISDICTION_BOUND" if jurisdiction_bound else "JURISDICTION_NEUTRAL",
        obj.context_scenario.jurisdiction_code if jurisdiction_bound else None,
        evidence_ids,
        regulatory_ids if jurisdiction_bound else (),
        non_jurisdiction_context,
    )


def build_context_directional_grounding_contract(
    *,
    decision_object: ContextualDecisionObject | dict[str, Any],
) -> dict[str, Any]:
    """Expose exact lens/scope grounding eligibility without adding evidence.

    This is deterministic metadata derived only from the frozen
    DecisionEvidencePack.  It helps the model use the same slot contract that
    the semantic validator enforces and prevents repair-only discovery of the
    allowed IDs.
    """

    obj = (
        decision_object
        if isinstance(decision_object, ContextualDecisionObject)
        else ContextualDecisionObject.model_validate(decision_object)
    )
    evidence_slots = {
        evidence_id: set(slots)
        for evidence_id, slots in obj.decision_evidence_pack.slot_coverage.items()
    }
    contract: dict[str, Any] = {}
    for lens_type in LensType:
        scope_contract: dict[str, Any] = {}
        for scope in LensClaimScope:
            allowed_slots = contextual_directional_slots(lens_type, scope)
            eligible = {
                evidence_id: sorted(slots & allowed_slots)
                for evidence_id, slots in evidence_slots.items()
                if slots & allowed_slots
            }
            scope_contract[scope.value] = {
                "eligible_slots": sorted(allowed_slots),
                "eligible_evidence": eligible,
                "support_slots": sorted(
                    contextual_directional_slots(
                        lens_type, scope, LensClaimStatus.SUPPORTS_FEASIBILITY
                    )
                ),
                "challenge_slots": sorted(
                    contextual_directional_slots(
                        lens_type, scope, LensClaimStatus.CHALLENGES_FEASIBILITY
                    )
                ),
                "required_scope_slots": sorted(
                    contextual_required_scope_slots(lens_type, scope)
                ),
            }
        contract[lens_type.value] = scope_contract
    return contract


class ContextLensAssessment(LensAssessment):
    """Contextual-only normalization for residual deployment-context gaps.

    The base Stage 4 v1 contract remains unchanged. In contextual v4, a lens
    may support GENERAL feasibility while still lacking deployment-specific
    details; that combination is PARTIAL rather than INSUFFICIENT_CONTEXT.
    """

    @model_validator(mode="before")
    @classmethod
    def normalize_partial_general_context_judgment(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        stance = value.get("stance")
        if stance not in (-1, 1) or value.get("evidence_sufficiency") != "INSUFFICIENT_CONTEXT":
            return value
        expected_status = "SUPPORTS_FEASIBILITY" if stance == 1 else "CHALLENGES_FEASIBILITY"
        matching = [
            claim
            for claim in (value.get("claims") or [])
            if isinstance(claim, dict) and claim.get("status") == expected_status
        ]
        if matching and all(claim.get("scope") == "GENERAL" for claim in matching):
            normalized = dict(value)
            normalized["evidence_sufficiency"] = "PARTIAL"
            return normalized
        return value


def build_constrained_context_lens_response_schema(
    *,
    decision_object: ContextualDecisionObject | dict[str, Any],
    expected_lens_type: LensType,
) -> dict[str, Any]:
    obj = (
        decision_object
        if isinstance(decision_object, ContextualDecisionObject)
        else ContextualDecisionObject.model_validate(decision_object)
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
        "enum": sorted(obj.evaluation_evidence_ids),
    }
    # Pydantic default_factory fields are optional in the generated JSON Schema,
    # but the contextual Stage 4 semantic contract expects the model to return
    # the complete structured object. In particular, PARTIAL/SUFFICIENT outputs
    # require at least one evidence-backed claim. Make that requirement visible
    # to llama.cpp constrained decoding instead of discovering an omitted
    # `claims` field only after generation in the Pydantic validator.
    schema["required"] = sorted(schema["properties"])
    schema["properties"]["claims"]["minItems"] = 1
    return schema


def validate_context_lens_assessment(
    assessment: LensAssessment | dict[str, Any],
    *,
    decision_object: ContextualDecisionObject | dict[str, Any],
    expected_lens_type: LensType,
) -> LensAssessment:
    try:
        parsed = (
            assessment
            if isinstance(assessment, ContextLensAssessment)
            else ContextLensAssessment.model_validate(
                assessment.model_dump(mode="json") if isinstance(assessment, LensAssessment) else assessment
            )
        )
        obj = (
            decision_object
            if isinstance(decision_object, ContextualDecisionObject)
            else ContextualDecisionObject.model_validate(decision_object)
        )
    except Exception as exc:
        raise Stage4ValidationError(f"Invalid contextual Stage 4 lens schema: {exc}") from exc

    errors: list[str] = []
    if parsed.decision_object_id != obj.decision_object_id:
        errors.append("decision_object_id mismatch")
    if parsed.lens_type != expected_lens_type:
        errors.append("lens_type mismatch")
    referenced = {
        evidence_id
        for claim in parsed.claims
        for evidence_id in claim.evidence_ids
    }
    outside = sorted(referenced - set(obj.evaluation_evidence_ids))
    if outside:
        errors.append(f"evidence outside contextual evaluation universe: {outside}")
    if any(value in (None, "") for value in (
        obj.deployment_context.region,
        obj.deployment_context.organization_type,
        obj.deployment_context.infrastructure_context,
        obj.deployment_context.budget_context,
    )):
        errors.append("main contextual Stage 4 requires all deployment context fields")
    if parsed.evidence_sufficiency == LensEvidenceSufficiency.INSUFFICIENT_CONTEXT:
        # Contextual runs may still identify a narrower missing datum, but the
        # four core deployment dimensions themselves are no longer absent.
        if not (parsed.evidence_gaps or parsed.conditional_requirements):
            errors.append("INSUFFICIENT_CONTEXT requires an explicit residual context gap")
    deployment_claims = [claim for claim in parsed.claims if claim.scope == LensClaimScope.DEPLOYMENT_SPECIFIC]
    if deployment_claims and not obj.context_evidence_ids:
        errors.append("DEPLOYMENT_SPECIFIC claims require frozen context evidence")

    slot_coverage = obj.decision_evidence_pack.slot_coverage
    evidence_slots = {
        evidence_id: set(slots)
        for evidence_id, slots in slot_coverage.items()
    }
    directional_claims = [
        claim
        for claim in parsed.claims
        if claim.status.value in {"SUPPORTS_FEASIBILITY", "CHALLENGES_FEASIBILITY"}
    ]
    downgraded_claim_ids: set[str] = set()
    for claim in directional_claims:
        direction_carrier_slots = contextual_directional_slots(
            expected_lens_type,
            claim.scope,
            claim.status,
        )
        required_scope_slots = contextual_required_scope_slots(
            expected_lens_type,
            claim.scope,
        )
        grounded_slots = {
            slot
            for evidence_id in claim.evidence_ids
            for slot in evidence_slots.get(evidence_id, set())
        }
        if not (grounded_slots & direction_carrier_slots) or not required_scope_slots.issubset(
            grounded_slots
        ):
            downgraded_claim_ids.add(claim.claim_id)
    if errors:
        raise Stage4ValidationError("; ".join(errors))

    # Conservative deterministic grounding canonicalization: a claim that is
    # factually cited but lacks a lens-eligible decision-evidence slot is not
    # allowed to carry feasibility direction. We only downgrade direction to
    # NEUTRAL_CONTEXT; we never add evidence or upgrade a stance.
    normalized = parsed.model_dump(mode="json")
    for claim in normalized["claims"]:
        if claim["claim_id"] in downgraded_claim_ids:
            claim["status"] = LensClaimStatus.NEUTRAL_CONTEXT.value

    support_ids = {
        claim["claim_id"]
        for claim in normalized["claims"]
        if claim["status"] == LensClaimStatus.SUPPORTS_FEASIBILITY.value
    }
    challenge_ids = {
        claim["claim_id"]
        for claim in normalized["claims"]
        if claim["status"] == LensClaimStatus.CHALLENGES_FEASIBILITY.value
    }

    # Conservative one-way stance canonicalization is retained: validation may
    # remove an unsupported nonzero stance, but must never manufacture or
    # strengthen a stance from 0 to +/-1. v4 instead rejects a logically
    # inconsistent neutral stance after strict grounding so the fixed semantic
    # repair turn must make the substantive judgment explicit.
    if normalized["stance"] == 1 and not support_ids:
        normalized["stance"] = 0
    elif normalized["stance"] == -1 and not challenge_ids:
        normalized["stance"] = 0

    if normalized["stance"] in (-1, 1) and normalized["evidence_sufficiency"] in {
        LensEvidenceSufficiency.INSUFFICIENT_EVIDENCE.value,
        LensEvidenceSufficiency.INSUFFICIENT_CONTEXT.value,
    }:
        normalized["evidence_sufficiency"] = LensEvidenceSufficiency.PARTIAL.value
    if normalized["evidence_sufficiency"] == LensEvidenceSufficiency.SUFFICIENT.value:
        # A grounding downgrade means at least one claimed directional basis
        # did not survive validation, so SUFFICIENT is no longer defensible.
        if downgraded_claim_ids:
            normalized["evidence_sufficiency"] = LensEvidenceSufficiency.PARTIAL.value

    reparsed = ContextLensAssessment.model_validate(normalized)

    valid_directional_claims = [
        claim
        for claim in reparsed.claims
        if claim.status in {
            LensClaimStatus.SUPPORTS_FEASIBILITY,
            LensClaimStatus.CHALLENGES_FEASIBILITY,
        }
    ]
    support_ids = {
        claim.claim_id
        for claim in reparsed.claims
        if claim.status == LensClaimStatus.SUPPORTS_FEASIBILITY
    }
    challenge_ids = {
        claim.claim_id
        for claim in reparsed.claims
        if claim.status == LensClaimStatus.CHALLENGES_FEASIBILITY
    }

    consistency_errors: list[str] = []
    if (
        reparsed.evidence_sufficiency == LensEvidenceSufficiency.INSUFFICIENT_CONTEXT
        and valid_directional_claims
        and all(claim.scope == LensClaimScope.GENERAL for claim in valid_directional_claims)
    ):
        consistency_errors.append(
            "GENERAL directional evidence survived strict slot grounding, so residual deployment-specific context "
            "must be represented as PARTIAL rather than INSUFFICIENT_CONTEXT; do not add evidence or invent a "
            "deployment-specific claim"
        )
    if reparsed.stance == 0 and bool(support_ids) != bool(challenge_ids):
        direction = "+1" if support_ids else "-1"
        claim_ids = sorted(support_ids or challenge_ids)
        consistency_errors.append(
            "stance=0 is inconsistent with one-sided grounded directional claims after validation: "
            f"expected direction {direction} from claim_ids={claim_ids}. Either use the matching nonzero stance "
            "with SUFFICIENT/PARTIAL evidence_sufficiency, or reclassify those claims as NEUTRAL_CONTEXT if the "
            "evidence does not actually justify direction. Do not add evidence solely to preserve stance=0."
        )
    if (
        reparsed.stance == 0
        and reparsed.evidence_sufficiency in {
            LensEvidenceSufficiency.PARTIAL,
            LensEvidenceSufficiency.SUFFICIENT,
        }
        and not support_ids
        and not challenge_ids
    ):
        consistency_errors.append(
            "stance=0 with PARTIAL/SUFFICIENT evidence cannot leave every feasibility claim NEUTRAL_CONTEXT. "
            "If the evidence is genuinely mixed, include at least one grounded SUPPORTS_FEASIBILITY claim and at "
            "least one grounded CHALLENGES_FEASIBILITY claim and explain why neither direction dominates. If no "
            "directional claim is justified, use INSUFFICIENT_EVIDENCE/INSUFFICIENT_CONTEXT instead. Do not add "
            "evidence or invent direction solely to satisfy this rule."
        )
    if consistency_errors:
        raise Stage4ValidationError("; ".join(consistency_errors))

    return reparsed


def validate_context_stage4_bundle(
    bundle: Stage4EvaluationBundle | dict[str, Any],
    *,
    decision_objects: list[ContextualDecisionObject | dict[str, Any]],
) -> Stage4EvaluationBundle:
    try:
        parsed = bundle if isinstance(bundle, Stage4EvaluationBundle) else Stage4EvaluationBundle.model_validate(bundle)
        objects = [
            item if isinstance(item, ContextualDecisionObject) else ContextualDecisionObject.model_validate(item)
            for item in decision_objects
        ]
    except Exception as exc:
        raise Stage4ValidationError(f"Invalid contextual Stage 4 bundle: {exc}") from exc
    if not objects:
        raise Stage4ValidationError("Contextual Stage 4 requires DecisionObjects")
    if parsed.case_id != objects[0].case_id or any(item.case_id != parsed.case_id for item in objects):
        raise Stage4ValidationError("Contextual Stage 4 case_id mismatch")
    expected_ids = {item.decision_object_id for item in objects}
    actual_ids = {item.decision_object_id for item in parsed.evaluations}
    if expected_ids != actual_ids or len(parsed.evaluations) != len(objects):
        raise Stage4ValidationError("Contextual Stage 4 requires one three-lens evaluation per scenario object")
    by_id = {item.decision_object_id: item for item in objects}
    for evaluation in parsed.evaluations:
        obj = by_id[evaluation.decision_object_id]
        validate_context_lens_assessment(
            evaluation.technical_feasibility,
            decision_object=obj,
            expected_lens_type=LensType.TECHNICAL_FEASIBILITY,
        )
        validate_context_lens_assessment(
            evaluation.institutional_regional,
            decision_object=obj,
            expected_lens_type=LensType.INSTITUTIONAL_REGIONAL,
        )
        validate_context_lens_assessment(
            evaluation.financial_adoption,
            decision_object=obj,
            expected_lens_type=LensType.FINANCIAL_ADOPTION,
        )

    # Jurisdiction names are scenario parameters, not factual evidence. If two
    # scenarios are in the same deterministic lens-equivalence class, their
    # overall feasibility direction must not diverge solely because the region
    # label changed. Institutional regulatory_applicability evidence makes the
    # equivalence key jurisdiction-bound and therefore permits different results.
    lens_fields = {
        LensType.TECHNICAL_FEASIBILITY: "technical_feasibility",
        LensType.INSTITUTIONAL_REGIONAL: "institutional_regional",
        LensType.FINANCIAL_ADOPTION: "financial_adoption",
    }
    for lens_type, field_name in lens_fields.items():
        grouped: dict[tuple[Any, ...], list[tuple[str, int]]] = {}
        for evaluation in parsed.evaluations:
            obj = by_id[evaluation.decision_object_id]
            signature = context_lens_equivalence_key(obj, lens_type)
            assessment = getattr(evaluation, field_name)
            grouped.setdefault(signature, []).append((obj.scenario_id, assessment.stance))
        for signature, members in grouped.items():
            stances = {stance for _, stance in members}
            if len(members) > 1 and len(stances) > 1:
                details = ", ".join(f"{scenario_id}:{stance:+d}" for scenario_id, stance in members)
                raise Stage4ValidationError(
                    "Jurisdiction-label sensitivity detected without lens-specific jurisdiction evidence: "
                    f"lens={lens_type.value}, scenarios={details}, equivalence_key={signature}. "
                    "A region label alone cannot change the GENERAL feasibility direction."
                )
    return parsed


__all__ = [
    "ContextLensAssessment",
    "DecisionLensEvaluation",
    "LensAssessment",
    "LensType",
    "Stage4EvaluationBundle",
    "STAGE4_CONTEXT_SEMANTIC_VALIDATION_VERSION",
    "build_context_directional_grounding_contract",
    "build_constrained_context_lens_response_schema",
    "context_lens_equivalence_key",
    "contextual_general_evidence_signature",
    "contextual_directional_slots",
    "contextual_required_scope_slots",
    "validate_context_lens_assessment",
    "validate_context_stage4_bundle",
]

