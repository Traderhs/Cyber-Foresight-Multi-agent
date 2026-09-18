from __future__ import annotations

import hashlib
from typing import Any

from Stage3.context_schema import ContextualDecisionObject
from Stage4.context_schema import LensType, build_context_directional_grounding_contract
from Stage4.schema import Stage4ValidationError


_PROMPT_EVIDENCE_FIELDS = (
    "evidence_id",
    "content",
    "source",
    "source_registry_id",
    "source_document_id",
    "available_at",
    "evidence_type",
    "retrieval_role",
    "threat_ids",
    "pmt_ids",
)


def _prompt_evidence_record(record: dict[str, Any]) -> dict[str, Any]:
    """Project a frozen evidence record to fields that can affect model judgment.

    Hashes, snapshot IDs, claim allow/deny lists, chain IDs, and other
    deterministic validation metadata remain in the frozen Stage 3 artifact.
    They are not repeated in the LLM prompt because Python already enforces
    those contracts before and after generation.
    """

    return {
        key: record[key]
        for key in _PROMPT_EVIDENCE_FIELDS
        if key in record and record[key] not in (None, "", [], {})
    }


def _prompt_decision_object(obj: ContextualDecisionObject) -> dict[str, Any]:
    """Return the non-duplicative Stage 4 prompt projection of a DecisionObject."""

    scenario = obj.context_scenario
    return {
        "decision_object_id": obj.decision_object_id,
        "case_id": obj.case_id,
        "threat": obj.threat,
        "threat_id": obj.threat_id,
        "pmt_or_mitigation": obj.pmt_or_mitigation,
        "pmt_or_mitigation_id": obj.pmt_or_mitigation_id,
        "forecast_interpretation": obj.forecast_interpretation,
        # Stage 2 remains reasoning provenance. Keep the full frozen summary,
        # but do not duplicate any evidence records around it.
        "stage2_critique_summary": obj.stage2_critique_summary.model_dump(mode="json"),
        "candidate_action": obj.candidate_action.model_dump(mode="json"),
        "intended_goal": obj.intended_goal,
        "timing_window": obj.timing_window.model_dump(mode="json"),
        "deployment_context": obj.deployment_context.model_dump(mode="json"),
        "known_constraints": obj.known_constraints,
        "unresolved_evidence_gaps": [
            item.model_dump(mode="json") for item in obj.unresolved_evidence_gaps
        ],
        "predictive_uncertainty": obj.predictive_uncertainty,
        "scenario": {
            "scenario_id": obj.scenario_id,
            "jurisdiction_code": scenario.jurisdiction_code,
            "region": scenario.region.value,
            "reference_enterprise_profile": scenario.reference_enterprise_profile.value,
            "assumptions": scenario.assumptions,
            "non_assumptions": scenario.non_assumptions,
        },
    }


def build_context_stage4_evaluation_payload(
    *,
    decision_object: ContextualDecisionObject | dict[str, Any],
    expected_lens_type: LensType | None = None,
) -> dict[str, Any]:
    obj = (
        decision_object
        if isinstance(decision_object, ContextualDecisionObject)
        else ContextualDecisionObject.model_validate(decision_object)
    )
    records = {
        str(record.get("evidence_id")): record
        for record in obj.evaluation_evidence_records
        if record.get("evidence_id")
    }
    expected_ids = sorted(obj.evaluation_evidence_ids)
    if set(records) != set(expected_ids):
        raise Stage4ValidationError(
            "Contextual Stage 4 requires evaluation_evidence_records to exactly match evaluation_evidence_ids"
        )
    full_grounding_contract = build_context_directional_grounding_contract(
        decision_object=obj
    )
    grounding_contract = (
        full_grounding_contract
        if expected_lens_type is None
        else {
            expected_lens_type.value: full_grounding_contract[expected_lens_type.value]
        }
    )
    # Avoid stable lexical/ID ordering becoming an accidental stance cue
    # (for example CHALLENGE records sorting before SUPPORT records).  The
    # order is deterministic and case-fixed, but blind to lens, jurisdiction,
    # slot, polarity, score, and model output.
    prompt_ids = sorted(
        expected_ids,
        key=lambda evidence_id: hashlib.sha256(
            f"stage4-evidence-order-v1|{obj.case_id}|{evidence_id}".encode("utf-8")
        ).hexdigest(),
    )
    return {
        # Do not serialize the full ContextualDecisionObject here. It embeds
        # evaluation_evidence_records and DecisionEvidencePack records that are
        # repeated below and previously caused 2-3 copies of the same evidence
        # text to enter every prompt.
        "decision_object": _prompt_decision_object(obj),
        "evaluation_evidence": [
            _prompt_evidence_record(records[evidence_id]) for evidence_id in prompt_ids
        ],
        "evidence_boundary": {
            "stage2_reasoning_is_external_evidence": False,
            "context_scenario_is_observed_real_organization": False,
            "sibling_lens_outputs_visible": False,
            "other_scenario_outputs_visible": False,
        },
        "directional_grounding_contract": grounding_contract,
    }

