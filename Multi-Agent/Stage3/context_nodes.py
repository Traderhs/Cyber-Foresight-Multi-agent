from __future__ import annotations

from pathlib import Path

from Pipeline.state import AgentState
from Stage3.artifact import load_frozen_stage3_artifact
from Stage3.context_artifact import freeze_stage3_context_artifact, load_frozen_stage3_context_artifact
from Stage3.context_builder import build_contextual_stage3_bundle
from Stage3.nodes import _load_exact_stage2_artifact


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_exact_base_stage3_artifact(state: AgentState) -> dict:
    if not state.get("stage3_complete"):
        raise ValueError("Contextual Stage 3 requires base Stage 3 completion")
    if not state.get("evidence_pack") or not state.get("stage3_decision_bundle"):
        raise ValueError("Contextual Stage 3 requires frozen Stage 0 evidence and base Stage 3 bundle")
    stage2_artifact = _load_exact_stage2_artifact(state)
    artifact, _, path = load_frozen_stage3_artifact(
        project_root=_project_root(),
        evidence_pack=state["evidence_pack"],
        stage2_artifact=stage2_artifact,
        stage0_config=state.get("stage0_config") or {},
    )
    if artifact is None:
        raise ValueError("Contextual Stage 3 requires the exact frozen base Stage 3 artifact")
    state_path = state.get("stage3_artifact_path")
    if not state_path or Path(state_path).resolve() != path.resolve():
        raise ValueError("Contextual Stage 3 state path does not match the exact base Stage 3 artifact")
    if artifact.get("decision_bundle") != state.get("stage3_decision_bundle"):
        raise ValueError("Contextual Stage 3 state bundle does not match the exact base Stage 3 artifact")
    if artifact.get("decision_bundle", {}).get("decision_objects") != state.get("decision_objects"):
        raise ValueError("Contextual Stage 3 decision_objects drifted from the exact base Stage 3 artifact")
    return artifact


def stage3_contextualize_node(state: AgentState) -> AgentState:
    """Build/reuse the deterministic KR/EU/US CIS-IG2 contextual bundle."""

    base_artifact = _load_exact_base_stage3_artifact(state)
    evidence_pack = state["evidence_pack"]
    frozen, identity, path = load_frozen_stage3_context_artifact(
        project_root=_project_root(),
        evidence_pack=evidence_pack,
        base_stage3_artifact=base_artifact,
    )
    if frozen is not None:
        bundle = frozen["contextual_decision_bundle"]
        return {
            "stage3_context_complete": True,
            "stage3_context_artifact_reused": True,
            "stage3_context_input_fingerprint": identity["input_fingerprint"],
            "stage3_context_artifact_path": str(path),
            "stage3_contextual_decision_bundle": bundle,
            "contextual_decision_objects": bundle["contextual_decision_objects"],
            "messages": [f"**Stage 3 Context**: reused contextual DecisionObject artifact {path}."],
        }

    bundle = build_contextual_stage3_bundle(
        project_root=_project_root(),
        evidence_pack=evidence_pack,
        base_stage3_bundle=state["stage3_decision_bundle"],
        base_stage3_input_fingerprint=str(state["stage3_input_fingerprint"]),
    )
    artifact, artifact_path = freeze_stage3_context_artifact(
        project_root=_project_root(),
        evidence_pack=evidence_pack,
        base_stage3_artifact=base_artifact,
        contextual_decision_bundle=bundle,
    )
    output = artifact["contextual_decision_bundle"]
    return {
        "stage3_context_complete": True,
        "stage3_context_artifact_reused": False,
        "stage3_context_input_fingerprint": artifact["input_fingerprint"],
        "stage3_context_artifact_path": str(artifact_path),
        "stage3_contextual_decision_bundle": output,
        "contextual_decision_objects": output["contextual_decision_objects"],
        "messages": [
            f"**Stage 3 Context**: froze {len(output['contextual_decision_objects'])} scenario-conditioned DecisionObject(s) at {artifact_path}."
        ],
    }

