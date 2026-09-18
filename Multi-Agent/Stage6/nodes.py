from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from Pipeline.state import AgentState
from Stage3.nodes import _load_exact_stage2_artifact
from Stage4.context_nodes import _load_exact_stage3_context_artifact
from Stage5.artifact import load_frozen_stage5_artifact
from Stage5.nodes import _load_exact_stage4_context_artifact
from Stage6.artifact import (
    clear_stage6_checkpoints,
    freeze_stage6_artifact,
    freeze_stage6_checkpoint,
    load_frozen_stage6_artifact,
    load_stage6_checkpoint,
)
from Stage6.builder import build_stage6_bundle, build_stage6_frames
from Stage6.prompts import STAGE6_SYNTHESIS_SYSTEM_PROMPT, STAGE6_SYNTHESIS_USER_PROMPT
from Stage6.runtime import STAGE6_CLIENT_CONCURRENCY, get_synthesis_llm
from Stage6.schema import (
    SynthesisNarrative,
    build_constrained_synthesis_narrative_schema,
    validate_synthesis_narrative,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_exact_stage5_artifact(
    state: AgentState,
    *,
    stage2_artifact: dict[str, Any],
    stage3_context_artifact: dict[str, Any],
    stage4_context_artifact: dict[str, Any],
) -> dict[str, Any]:
    if not state.get("stage5_complete") or not state.get("stage5_diagnostic_bundle"):
        raise ValueError("Stage 6 requires Stage 5 completion and its deterministic diagnostic bundle")
    artifact, _, path = load_frozen_stage5_artifact(
        project_root=_project_root(),
        evidence_pack=state["evidence_pack"],
        stage2_artifact=stage2_artifact,
        stage3_context_artifact=stage3_context_artifact,
        stage4_context_artifact=stage4_context_artifact,
    )
    if artifact is None:
        raise ValueError("Stage 6 requires the exact frozen Stage 5 artifact")
    state_path = state.get("stage5_artifact_path")
    if not state_path or Path(state_path).resolve() != path.resolve():
        raise ValueError("Stage 6 state path does not match exact Stage 5 artifact")
    if artifact.get("diagnostic_bundle") != state.get("stage5_diagnostic_bundle"):
        raise ValueError("Stage 6 state Stage 5 bundle drifted from exact frozen artifact")
    return artifact


def _exact_upstream(state: AgentState) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    stage2_artifact = _load_exact_stage2_artifact(state)
    stage3_context_artifact = _load_exact_stage3_context_artifact(state)
    stage4_context_artifact = _load_exact_stage4_context_artifact(
        state,
        stage3_context_artifact=stage3_context_artifact,
    )
    stage5_artifact = _load_exact_stage5_artifact(
        state,
        stage2_artifact=stage2_artifact,
        stage3_context_artifact=stage3_context_artifact,
        stage4_context_artifact=stage4_context_artifact,
    )
    return stage3_context_artifact, stage4_context_artifact, stage5_artifact


def _frames(
    *,
    stage3_context_artifact: dict[str, Any],
    stage4_context_artifact: dict[str, Any],
    stage5_artifact: dict[str, Any],
) -> list[dict[str, Any]]:
    return build_stage6_frames(
        stage3_context_bundle=stage3_context_artifact["contextual_decision_bundle"],
        stage4_context_bundle=stage4_context_artifact["evaluation_bundle"],
        stage5_diagnostic_bundle=stage5_artifact["diagnostic_bundle"],
    )


def prepare_stage6_node(state: AgentState) -> AgentState:
    """Reuse a frozen Stage 6 artifact or load exact per-DecisionObject checkpoints."""

    stage3, stage4, stage5 = _exact_upstream(state)
    frozen, identity, path = load_frozen_stage6_artifact(
        project_root=_project_root(),
        stage3_context_artifact=stage3,
        stage4_context_artifact=stage4,
        stage5_artifact=stage5,
    )
    if frozen is not None:
        bundle = frozen["synthesis_bundle"]
        return {
            "stage6_complete": False,
            "stage6_artifact_reused": True,
            "stage6_input_fingerprint": identity["input_fingerprint"],
            "stage6_artifact_path": str(path),
            "stage6_synthesis_bundle": bundle,
            "decision_syntheses": bundle["syntheses"],
            "stage6_narratives": [],
            "stage6_checkpoint_reuse_count": 0,
            "messages": [f"**Stage 6**: reused frozen synthesis artifact {path}."],
        }

    narratives: list[dict[str, Any]] = []
    for frame in _frames(
        stage3_context_artifact=stage3,
        stage4_context_artifact=stage4,
        stage5_artifact=stage5,
    ):
        decision_object_id = frame["deterministic_output"]["decision_object_id"]
        checkpoint, _ = load_stage6_checkpoint(
            project_root=_project_root(),
            stage3_context_artifact=stage3,
            stage4_context_artifact=stage4,
            stage5_artifact=stage5,
            decision_object_id=decision_object_id,
        )
        if checkpoint is not None:
            narratives.append(checkpoint["narrative"])
    return {
        "stage6_complete": False,
        "stage6_artifact_reused": False,
        "stage6_input_fingerprint": identity["input_fingerprint"],
        "stage6_artifact_path": str(path),
        "stage6_synthesis_bundle": None,
        "decision_syntheses": [],
        "stage6_narratives": narratives,
        "stage6_checkpoint_reuse_count": len(narratives),
        "messages": [
            f"**Stage 6**: loaded {len(narratives)} exact synthesis checkpoint(s); missing DecisionObjects will generate."
        ],
    }


async def _run_synthesis(
    *,
    frame: dict[str, Any],
    stage3_context_artifact: dict[str, Any],
    stage4_context_artifact: dict[str, Any],
    stage5_artifact: dict[str, Any],
    scenario_index: int,
    scenario_count: int,
) -> dict[str, Any]:
    frozen = frame["deterministic_output"]
    response_schema = build_constrained_synthesis_narrative_schema(frame=frame)
    structured_llm = get_synthesis_llm().with_structured_output(
        SynthesisNarrative,
        method="json_schema",
        progress_label=(
            f"Stage6-Synthesis-{scenario_index}of{scenario_count}-{frozen['scenario_id']}"
        ),
        response_schema=response_schema,
        semantic_validator=lambda narrative: validate_synthesis_narrative(narrative, frame=frame),
    )
    raw = await structured_llm.ainvoke(
        [
            SystemMessage(content=STAGE6_SYNTHESIS_SYSTEM_PROMPT),
            HumanMessage(
                content=STAGE6_SYNTHESIS_USER_PROMPT.format(
                    synthesis_frame=json.dumps(
                        {
                            key: value
                            for key, value in frame.items()
                            if key != "validation_context"
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
            ),
        ]
    )
    narrative = raw.model_dump(mode="json")
    checkpoint_path = freeze_stage6_checkpoint(
        project_root=_project_root(),
        stage3_context_artifact=stage3_context_artifact,
        stage4_context_artifact=stage4_context_artifact,
        stage5_artifact=stage5_artifact,
        decision_object_id=frozen["decision_object_id"],
        narrative=narrative,
    )
    print(
        f"Stage 6 {frozen['scenario_id']} checkpoint saved: {checkpoint_path}",
        flush=True,
    )
    return narrative


async def stage6_synthesis_node(state: AgentState) -> AgentState:
    """Generate one constrained synthesis per scenario with exactly two workers."""

    if state.get("stage6_artifact_reused"):
        return {"messages": ["**Stage 6**: reused frozen synthesis outputs."]}
    stage3, stage4, stage5 = _exact_upstream(state)
    frames = _frames(
        stage3_context_artifact=stage3,
        stage4_context_artifact=stage4,
        stage5_artifact=stage5,
    )
    existing = {
        str(item["decision_object_id"]): item
        for item in (state.get("stage6_narratives") or [])
    }
    outputs: dict[str, dict[str, Any]] = {}
    pending: asyncio.Queue[tuple[int, dict[str, Any]]] = asyncio.Queue()
    for index, frame in enumerate(frames, start=1):
        decision_object_id = frame["deterministic_output"]["decision_object_id"]
        cached = existing.get(decision_object_id)
        if cached is not None:
            parsed = validate_synthesis_narrative(cached, frame=frame)
            outputs[decision_object_id] = parsed.model_dump(mode="json")
        else:
            pending.put_nowait((index, frame))

    print(
        f"Stage 6 synthesis queue: {pending.qsize()} missing of {len(frames)} scenario syntheses; "
        f"{STAGE6_CLIENT_CONCURRENCY} workers share the existing two-slot llama.cpp runtime.",
        flush=True,
    )

    async def worker(worker_index: int) -> None:
        while True:
            try:
                scenario_index, frame = pending.get_nowait()
            except asyncio.QueueEmpty:
                return
            frozen = frame["deterministic_output"]
            try:
                print(
                    f"Stage 6 worker {worker_index}: starting {frozen['scenario_id']} synthesis",
                    flush=True,
                )
                narrative = await _run_synthesis(
                    frame=frame,
                    stage3_context_artifact=stage3,
                    stage4_context_artifact=stage4,
                    stage5_artifact=stage5,
                    scenario_index=scenario_index,
                    scenario_count=len(frames),
                )
                outputs[frozen["decision_object_id"]] = narrative
            finally:
                pending.task_done()

    await asyncio.gather(
        *[worker(index + 1) for index in range(STAGE6_CLIENT_CONCURRENCY)]
    )
    narratives = [
        outputs[frame["deterministic_output"]["decision_object_id"]]
        for frame in frames
    ]
    return {
        "stage6_narratives": narratives,
        "messages": [
            f"**Stage 6**: completed {len(narratives)} constrained scenario synthesis narrative(s) with two-worker scheduling."
        ],
    }


def stage6_complete_node(state: AgentState) -> AgentState:
    if state.get("stage6_artifact_reused"):
        if not state.get("stage6_synthesis_bundle"):
            raise ValueError("Reused Stage 6 artifact is missing its synthesis bundle")
        return {
            "stage6_complete": True,
            "messages": [f"**Stage 6**: reused immutable artifact {state.get('stage6_artifact_path')}."],
        }

    stage3, stage4, stage5 = _exact_upstream(state)
    frames = _frames(
        stage3_context_artifact=stage3,
        stage4_context_artifact=stage4,
        stage5_artifact=stage5,
    )
    narratives = state.get("stage6_narratives") or []
    if len(narratives) != len(frames):
        raise ValueError("Stage 6 requires exactly one synthesis narrative per contextual DecisionObject")
    bundle = build_stage6_bundle(
        case_id=str(stage5["case_id"]),
        frames=frames,
        narratives=narratives,
    )
    artifact, path = freeze_stage6_artifact(
        project_root=_project_root(),
        stage3_context_artifact=stage3,
        stage4_context_artifact=stage4,
        stage5_artifact=stage5,
        synthesis_bundle=bundle,
    )
    clear_stage6_checkpoints(
        project_root=_project_root(),
        stage3_context_artifact=stage3,
        stage4_context_artifact=stage4,
        stage5_artifact=stage5,
    )
    output = artifact["synthesis_bundle"]
    return {
        "stage6_complete": True,
        "stage6_artifact_reused": False,
        "stage6_input_fingerprint": artifact["input_fingerprint"],
        "stage6_artifact_path": str(path),
        "stage6_synthesis_bundle": output,
        "decision_syntheses": output["syntheses"],
        "messages": [
            f"**Stage 6**: froze {len(output['syntheses'])} disagreement-preserving synthesis(es) at {path}."
        ],
    }

