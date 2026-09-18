from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .builder import Stage0Builder
from .schema import EvaluationMode


def _prompt_feature_view(features: dict[str, Any]) -> dict[str, Any]:
    return {
        name: {
            "available": value["available"],
            "recent_direction": value["recent_direction"],
            "latest_value_z": value["latest_value_z"],
            "latest_field_id": (
                value["historical_field_ids"][-1]
                if value["available"] and value["historical_field_ids"]
                else None
            ),
        }
        for name, value in features.items()
    }


def _prompt_state_view(state: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in state.items() if key != "forecast_field_ids"}


def _prompt_retrieval_summary(metadata: dict[str, Any]) -> dict[str, Any]:
    """Expose only decision-relevant retrieval facts to the critic prompt.

    Full BM25 queries, candidate score tables, and slot-level audit traces remain
    in the frozen EvidencePack for reproducibility but are not evidence and must
    not consume the critic's reasoning context.
    """
    keys = (
        "cutoff_applied",
        "retrieved_count",
        "source_family_count",
        "independent_evidence_chain_count",
        "required_evidence_slots",
        "covered_evidence_slots",
        "missing_evidence_slots",
        "evidence_sufficiency",
        "temporal_violation_count",
        "source_contract_violation_count",
    )
    return {key: metadata[key] for key in keys if key in metadata}


def build_agent_input(
    *,
    project_root: str | Path,
    snapshot_id: str,
    threat: str,
    pmt: str,
    analysis_cutoff_date: str,
    evaluation_mode: EvaluationMode | str,
    case_id: str | None = None,
) -> dict[str, Any]:
    """Build the Agent-facing Stage 0 payload from existing experiment artifacts only."""
    root = Path(project_root).resolve()
    builder = Stage0Builder(
        root / "Multi-Agent/Results/Stage0/Forecast",
        root / "Multi-Agent/Results/Stage0/Evidence/snapshots" / snapshot_id,
    )
    pack = builder.build(
        case_id=case_id or f"{threat}__{pmt}",
        threat=threat,
        pmt=pmt,
        analysis_cutoff_date=analysis_cutoff_date,
        evaluation_mode=evaluation_mode,
    )
    pack_dict = pack.to_dict()
    summary = pack_dict["forecast_summary"]
    slot_coverage = pack_dict["retrieval_metadata"]["evidence_slot_coverage"]
    prompt_payload = {
        "case_id": pack_dict["case_id"],
        "threat_id": pack_dict["threat_id"],
        "pmt_id": pack_dict["pmt_id"],
        "forecast_origin_date": pack_dict["forecast_origin_date"],
        "training_data_end": pack_dict["training_data_end"],
        "analysis_cutoff_date": pack_dict["analysis_cutoff_date"],
        "evaluation_mode": pack_dict["evaluation_mode"],
        "forecast_summary": {
            "paper_feature_order": summary["paper_feature_order"],
            "historical_input": {
                "threat_features": _prompt_feature_view(summary["historical_input"]["threat_features"]),
                "pmt_features": _prompt_feature_view(summary["historical_input"]["pmt_features"]),
            },
            "threat_state": _prompt_state_view(summary["threat_state"]),
            "pmt_state": _prompt_state_view(summary["pmt_state"]),
            "gap_by_year": summary["gap_by_year"],
            "gap_slope_per_year": summary["gap_slope_per_year"],
            "gap_direction": summary["gap_direction"],
            "gap_semantics": summary["gap_semantics"],
            "gap_direction_semantics": summary.get("gap_direction_semantics"),
            "predictive_uncertainty": summary["predictive_uncertainty"],
            "forecast_field_ids": summary["forecast_field_ids"],
        },
        "evidence": [
            {
                "evidence_id": record["evidence_id"],
                "source": record["source"],
                "source_family": record["source_family"],
                "publication_date": record["publication_date"],
                "evidence_date": record["evidence_date"],
                "available_at": record["available_at"],
                "source_document_id": record["source_document_id"],
                "source_snapshot_id": record["source_snapshot_id"],
                "source_locator": record["source_locator"],
                "evidence_chain_id": record["evidence_chain_id"],
                "evidence_type": record["evidence_type"],
                "retrieval_role": record["retrieval_role"],
                "retrieval_slots": slot_coverage.get(record["evidence_id"], []),
                "allowed_claim_types": record["allowed_claim_types"],
                "prohibited_claim_types": record["prohibited_claim_types"],
                "content": record["content"],
            }
            for record in pack_dict["evidence"]
        ],
        "retrieval_summary": _prompt_retrieval_summary(pack_dict["retrieval_metadata"]),
    }
    return {
        "forecast_data": json.dumps(prompt_payload, ensure_ascii=False, indent=2),
        "evidence_pack": pack_dict,
    }
