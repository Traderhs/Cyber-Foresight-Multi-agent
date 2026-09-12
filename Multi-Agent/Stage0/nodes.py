from __future__ import annotations

import os
from pathlib import Path

from Pipeline.state import AgentState
from Stage0.agent_input import build_agent_input
from Stage0.schema import EvaluationMode


def load_data_node(state: AgentState) -> AgentState:
    """Load one canonical Stage 0 Forecast + EvidencePack into pipeline state."""
    explicit = state.get("stage0_config") or {}
    required = {
        "snapshot_id": explicit.get("snapshot_id") or os.getenv("STAGE0_SNAPSHOT_ID"),
        "threat": explicit.get("threat") or os.getenv("STAGE0_THREAT"),
        "pmt": explicit.get("pmt") or os.getenv("STAGE0_PMT"),
        "analysis_cutoff_date": explicit.get("analysis_cutoff_date") or os.getenv("STAGE0_ANALYSIS_CUTOFF"),
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(f"Missing required Stage 0 runtime configuration: {', '.join(missing)}")

    project_root = Path(__file__).resolve().parents[2]
    snapshot_id = required["snapshot_id"]
    threat = required["threat"]
    pmt = required["pmt"]
    analysis_cutoff = required["analysis_cutoff_date"]
    evaluation_mode = EvaluationMode(
        explicit.get("evaluation_mode")
        or os.getenv("STAGE0_EVALUATION_MODE", EvaluationMode.CURRENT_REASSESSMENT.value)
    )
    case_id = explicit.get("case_id") or os.getenv("STAGE0_CASE_ID", f"{threat}__{pmt}")

    agent_input = build_agent_input(
        project_root=project_root,
        snapshot_id=snapshot_id,
        threat=threat,
        pmt=pmt,
        analysis_cutoff_date=analysis_cutoff,
        evaluation_mode=evaluation_mode,
        case_id=case_id,
    )
    pack_dict = agent_input["evidence_pack"]
    retrieval = pack_dict["retrieval_metadata"]
    min_slots = explicit.get("min_covered_evidence_slots")
    min_families = explicit.get("min_source_families")
    if min_slots is not None and len(retrieval.get("covered_evidence_slots", [])) < int(min_slots):
        raise ValueError(
            f"Main-set evidence qualification failed for {case_id}: "
            f"covered_slots={len(retrieval.get('covered_evidence_slots', []))} < {min_slots}"
        )
    if min_families is not None and int(retrieval.get("source_family_count", 0)) < int(min_families):
        raise ValueError(
            f"Main-set evidence qualification failed for {case_id}: "
            f"source_family_count={retrieval.get('source_family_count', 0)} < {min_families}"
        )
    print(f"Stage 0 EvidencePack loaded: {case_id} ({len(pack_dict['evidence'])} evidence records)\n")
    return {
        "forecast_data": agent_input["forecast_data"],
        "evidence_pack": pack_dict,
        "iteration_count": 0,
        "messages": [],
    }
