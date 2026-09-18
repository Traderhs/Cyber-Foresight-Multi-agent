from __future__ import annotations

from pathlib import Path

from Pipeline.state import AgentState
from Stage1.artifact import load_frozen_stage1_artifact
from Stage2.artifact import load_frozen_stage2_artifact
from Stage3.artifact import freeze_stage3_artifact, load_frozen_stage3_artifact
from Stage3.builder import build_stage3_bundle


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_exact_stage2_artifact(state: AgentState) -> dict:
    if not state.get("evidence_pack") or not state.get("forecast_data"):
        raise ValueError("Stage 3 requires Stage 0 evidence/forecast state to validate Stage 2 provenance")
    if not state.get("attack_assessment") or not state.get("defense_assessment"):
        raise ValueError("Stage 3 requires the frozen Stage 1 pre-assessments")

    stage1_artifact, _, stage1_path = load_frozen_stage1_artifact(
        project_root=_project_root(),
        evidence_pack=state["evidence_pack"],
        forecast_data=state["forecast_data"],
    )
    if stage1_artifact is None:
        raise ValueError("Stage 3 cannot validate Stage 2 because the exact frozen Stage 1 artifact is missing")
    state_stage1_path = state.get("stage1_artifact_path")
    if state_stage1_path and Path(state_stage1_path).resolve() != stage1_path.resolve():
        raise ValueError("Stage 3 state Stage 1 artifact path does not match the exact frozen Stage 1 artifact")

    artifact, _, stage2_path = load_frozen_stage2_artifact(
        project_root=_project_root(),
        evidence_pack=state["evidence_pack"],
        forecast_data=state["forecast_data"],
        stage1_artifact=stage1_artifact,
        attack_pre_assessment=state["attack_assessment"],
        defense_pre_assessment=state["defense_assessment"],
    )
    if artifact is None:
        raise ValueError("Stage 3 requires the exact validated frozen Stage 2 artifact")
    path_value = state.get("stage2_artifact_path")
    if not path_value or Path(path_value).resolve() != stage2_path.resolve():
        raise ValueError("Stage 3 state Stage 2 artifact path does not match the exact frozen Stage 2 artifact")
    if artifact.get("debate_result") != state.get("stage2_debate_result"):
        raise ValueError("Stage 3 state Stage 2 result does not match the exact frozen Stage 2 artifact")
    return artifact


def stage3_build_node(state: AgentState) -> AgentState:
    """Build/freeze the deterministic Common Decision Object bundle."""
    if not state.get("stage2_complete"):
        raise ValueError("Stage 3 requires Stage 2 completion")
    if not state.get("evidence_pack") or not state.get("stage2_debate_result"):
        raise ValueError("Stage 3 requires Stage 0 EvidencePack and Stage 2 result")

    stage2_artifact = _load_exact_stage2_artifact(state)
    stage0_config = state.get("stage0_config") or {}
    frozen, identity, path = load_frozen_stage3_artifact(
        project_root=_project_root(),
        evidence_pack=state["evidence_pack"],
        stage2_artifact=stage2_artifact,
        stage0_config=stage0_config,
    )
    if frozen is not None:
        bundle = frozen["decision_bundle"]
        return {
            "stage3_complete": True,
            "stage3_artifact_reused": True,
            "stage3_input_fingerprint": identity["input_fingerprint"],
            "stage3_artifact_path": str(path),
            "stage3_decision_bundle": bundle,
            "decision_objects": bundle["decision_objects"],
            "messages": [f"**Stage 3**: reused deterministic Common Decision Object artifact {path}."],
        }

    bundle = build_stage3_bundle(
        evidence_pack=state["evidence_pack"],
        stage0_config=stage0_config,
        stage2_result=state["stage2_debate_result"],
    )
    artifact, artifact_path = freeze_stage3_artifact(
        project_root=_project_root(),
        evidence_pack=state["evidence_pack"],
        stage2_artifact=stage2_artifact,
        stage0_config=stage0_config,
        decision_bundle=bundle,
    )
    output = artifact["decision_bundle"]
    return {
        "stage3_complete": True,
        "stage3_artifact_reused": False,
        "stage3_input_fingerprint": artifact["input_fingerprint"],
        "stage3_artifact_path": str(artifact_path),
        "stage3_decision_bundle": output,
        "decision_objects": output["decision_objects"],
        "messages": [
            f"**Stage 3**: froze {len(output['decision_objects'])} Common Decision Object(s) at {artifact_path}."
        ],
    }
