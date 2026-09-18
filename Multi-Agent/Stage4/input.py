from __future__ import annotations

from typing import Any

from Stage3.schema import DecisionObject
from Stage4.schema import Stage4ValidationError


def build_stage4_evaluation_payload(
    *,
    decision_object: DecisionObject | dict[str, Any],
    evidence_pack: dict[str, Any],
) -> dict[str, Any]:
    """Materialize the identical evidence-bounded payload given to every Stage 4 lens."""

    obj = (
        decision_object
        if isinstance(decision_object, DecisionObject)
        else DecisionObject.model_validate(decision_object)
    )
    records = {
        str(record.get("evidence_id")): record
        for record in evidence_pack.get("evidence", [])
        if record.get("evidence_id")
    }
    expected_ids = list(obj.evaluation_evidence_ids)
    if set(expected_ids) != set(records):
        missing = sorted(set(expected_ids) - set(records))
        extra = sorted(set(records) - set(expected_ids))
        raise Stage4ValidationError(
            "Stage 4 requires the DecisionObject evidence universe to equal the frozen Stage 0 EvidencePack: "
            f"missing={missing}, extra={extra}"
        )

    return {
        "decision_object": obj.model_dump(mode="json"),
        "evaluation_evidence": [records[evidence_id] for evidence_id in sorted(expected_ids)],
        "evidence_boundary": {
            "allowed_evidence_ids": sorted(expected_ids),
            "stage2_reasoning_is_external_evidence": False,
            "sibling_lens_outputs_visible": False,
        },
    }

