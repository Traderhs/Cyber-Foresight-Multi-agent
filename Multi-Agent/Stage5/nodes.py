from __future__ import annotations

from pathlib import Path
from typing import Any

from Pipeline.state import AgentState
from Stage3.nodes import _load_exact_stage2_artifact
from Stage4.context_artifact import load_frozen_stage4_context_artifact
from Stage4.context_nodes import _load_exact_stage3_context_artifact
from Stage5.artifact import freeze_stage5_artifact, load_frozen_stage5_artifact
from Stage5.builder import build_stage5_diagnostic_bundle


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_exact_stage4_context_artifact(
    state: AgentState,
    *,
    stage3_context_artifact: dict[str, Any],
) -> dict[str, Any]:
    if not state.get("stage4_context_complete"):
        raise ValueError("Stage 5 requires contextual Stage 4 completion")
    if not state.get("stage4_context_evaluation_bundle"):
        raise ValueError("Stage 5 requires the contextual Stage 4 evaluation bundle")
    artifact, _, path = load_frozen_stage4_context_artifact(
        project_root=_project_root(),
        stage3_context_artifact=stage3_context_artifact,
    )
    if artifact is None:
        raise ValueError("Stage 5 requires the exact frozen contextual Stage 4 artifact")
    state_path = state.get("stage4_context_artifact_path")
    if not state_path or Path(state_path).resolve() != path.resolve():
        raise ValueError("Stage 5 state path does not match exact contextual Stage 4 artifact")
    if artifact.get("evaluation_bundle") != state.get("stage4_context_evaluation_bundle"):
        raise ValueError("Stage 5 state Stage 4 bundle drifted from exact frozen artifact")
    return artifact


def stage5_diagnostics_node(state: AgentState) -> AgentState:
    """Compute deterministic disagreement/evidence diagnostics; no LLM call is allowed."""

    if not state.get("evidence_pack") or not state.get("stage2_debate_result"):
        raise ValueError("Stage 5 requires Stage 0 EvidencePack and Stage 2 debate result")
    stage2_artifact = _load_exact_stage2_artifact(state)
    stage3_context_artifact = _load_exact_stage3_context_artifact(state)
    stage4_context_artifact = _load_exact_stage4_context_artifact(
        state,
        stage3_context_artifact=stage3_context_artifact,
    )

    frozen, identity, path = load_frozen_stage5_artifact(
        project_root=_project_root(),
        evidence_pack=state["evidence_pack"],
        stage2_artifact=stage2_artifact,
        stage3_context_artifact=stage3_context_artifact,
        stage4_context_artifact=stage4_context_artifact,
    )
    if frozen is not None:
        return {
            "stage5_complete": True,
            "stage5_artifact_reused": True,
            "stage5_input_fingerprint": identity["input_fingerprint"],
            "stage5_artifact_path": str(path),
            "stage5_diagnostic_bundle": frozen["diagnostic_bundle"],
            "messages": [f"**Stage 5**: reused deterministic diagnostic artifact {path}."],
        }

    bundle = build_stage5_diagnostic_bundle(
        evidence_pack=state["evidence_pack"],
        stage2_result=state["stage2_debate_result"],
        stage3_context_bundle=stage3_context_artifact["contextual_decision_bundle"],
        stage4_context_bundle=stage4_context_artifact["evaluation_bundle"],
    )
    artifact, artifact_path = freeze_stage5_artifact(
        project_root=_project_root(),
        evidence_pack=state["evidence_pack"],
        stage2_artifact=stage2_artifact,
        stage3_context_artifact=stage3_context_artifact,
        stage4_context_artifact=stage4_context_artifact,
        diagnostic_bundle=bundle,
    )
    return {
        "stage5_complete": True,
        "stage5_artifact_reused": False,
        "stage5_input_fingerprint": artifact["input_fingerprint"],
        "stage5_artifact_path": str(artifact_path),
        "stage5_diagnostic_bundle": artifact["diagnostic_bundle"],
        "messages": [
            f"**Stage 5**: froze deterministic disagreement/evidence diagnostics at {artifact_path}."
        ],
    }
