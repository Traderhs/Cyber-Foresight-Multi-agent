from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from Pipeline.state import AgentState
from Stage1.runtime import LlamaCppChatClient, assert_llama_server_ready
from Stage3.artifact import load_frozen_stage3_artifact
from Stage3.nodes import _load_exact_stage2_artifact
from Stage3.schema import DecisionObject
from Stage4.artifact import (
    clear_stage4_lens_checkpoints,
    freeze_stage4_artifact,
    freeze_stage4_lens_checkpoint,
    load_frozen_stage4_artifact,
    load_stage4_lens_checkpoint,
)
from Stage4.input import build_stage4_evaluation_payload
from Stage4.prompts import STAGE4_EVALUATION_USER_PROMPT, SYSTEM_PROMPT_BY_LENS
from Stage4.schema import (
    DecisionLensEvaluation,
    LensAssessment,
    LensType,
    Stage4EvaluationBundle,
    build_constrained_lens_response_schema,
    validate_lens_assessment,
    validate_stage4_evaluation_bundle,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_llm() -> LlamaCppChatClient:
    assert_llama_server_ready()
    return LlamaCppChatClient()


def _load_exact_stage3_artifact(state: AgentState) -> dict[str, Any]:
    if not state.get("stage3_complete"):
        raise ValueError("Stage 4 requires Stage 3 completion")
    if not state.get("evidence_pack") or not state.get("stage3_decision_bundle"):
        raise ValueError("Stage 4 requires the frozen Stage 0 EvidencePack and Stage 3 decision bundle")
    stage2_artifact = _load_exact_stage2_artifact(state)
    artifact, _, stage3_path = load_frozen_stage3_artifact(
        project_root=_project_root(),
        evidence_pack=state["evidence_pack"],
        stage2_artifact=stage2_artifact,
        stage0_config=state.get("stage0_config") or {},
    )
    if artifact is None:
        raise ValueError("Stage 4 requires the exact validated frozen Stage 3 artifact")
    path_value = state.get("stage3_artifact_path")
    if not path_value or Path(path_value).resolve() != stage3_path.resolve():
        raise ValueError("Stage 4 state Stage 3 artifact path does not match the exact frozen Stage 3 artifact")
    if artifact.get("decision_bundle") != state.get("stage3_decision_bundle"):
        raise ValueError("Stage 4 state Stage 3 bundle does not match the exact frozen Stage 3 artifact")
    frozen_decision_objects = (artifact.get("decision_bundle") or {}).get("decision_objects") or []
    if frozen_decision_objects != (state.get("decision_objects") or []):
        raise ValueError(
            "Stage 4 state decision_objects do not match the exact frozen Stage 3 artifact"
        )
    return artifact


def _assessments_from_frozen_bundle(bundle: dict[str, Any], lens_type: LensType) -> list[dict[str, Any]]:
    key_by_lens = {
        LensType.TECHNICAL_FEASIBILITY: "technical_feasibility",
        LensType.INSTITUTIONAL_REGIONAL: "institutional_regional",
        LensType.FINANCIAL_ADOPTION: "financial_adoption",
    }
    key = key_by_lens[lens_type]
    return [evaluation[key] for evaluation in bundle.get("evaluations", [])]


def prepare_stage4_node(state: AgentState) -> AgentState:
    """Resolve exact Stage 4 identity and reuse final/per-lens artifacts when available."""

    stage3_artifact = _load_exact_stage3_artifact(state)
    evidence_pack = state["evidence_pack"]
    frozen, identity, artifact_path = load_frozen_stage4_artifact(
        project_root=_project_root(),
        evidence_pack=evidence_pack,
        stage3_artifact=stage3_artifact,
    )
    if frozen is not None:
        evaluation_bundle = frozen["evaluation_bundle"]
        return {
            "stage4_complete": False,
            "stage4_artifact_reused": True,
            "stage4_input_fingerprint": identity["input_fingerprint"],
            "stage4_artifact_path": str(artifact_path),
            "stage4_evaluation_bundle": evaluation_bundle,
            "technical_lens_assessments": _assessments_from_frozen_bundle(
                evaluation_bundle, LensType.TECHNICAL_FEASIBILITY
            ),
            "institutional_lens_assessments": _assessments_from_frozen_bundle(
                evaluation_bundle, LensType.INSTITUTIONAL_REGIONAL
            ),
            "financial_lens_assessments": _assessments_from_frozen_bundle(
                evaluation_bundle, LensType.FINANCIAL_ADOPTION
            ),
            "technical_checkpoint_reuse_count": 0,
            "institutional_checkpoint_reuse_count": 0,
            "financial_checkpoint_reuse_count": 0,
            "messages": [f"**Stage 4**: reused frozen parallel-lens artifact {artifact_path}."],
        }

    decision_objects = [DecisionObject.model_validate(item) for item in state.get("decision_objects") or []]
    if not decision_objects:
        raise ValueError("Stage 4 preparation requires at least one frozen DecisionObject")
    checkpoint_outputs: dict[LensType, list[dict[str, Any]]] = {lens: [] for lens in LensType}
    for decision_object in decision_objects:
        for lens_type in LensType:
            checkpoint, _ = load_stage4_lens_checkpoint(
                project_root=_project_root(),
                evidence_pack=evidence_pack,
                stage3_artifact=stage3_artifact,
                decision_object_id=decision_object.decision_object_id,
                lens_type=lens_type,
            )
            if checkpoint is not None:
                checkpoint_outputs[lens_type].append(checkpoint["assessment"])

    return {
        "stage4_complete": False,
        "stage4_artifact_reused": False,
        "stage4_input_fingerprint": identity["input_fingerprint"],
        "stage4_artifact_path": str(artifact_path),
        "stage4_evaluation_bundle": None,
        "technical_lens_assessments": checkpoint_outputs[LensType.TECHNICAL_FEASIBILITY],
        "institutional_lens_assessments": checkpoint_outputs[LensType.INSTITUTIONAL_REGIONAL],
        "financial_lens_assessments": checkpoint_outputs[LensType.FINANCIAL_ADOPTION],
        "technical_checkpoint_reuse_count": len(checkpoint_outputs[LensType.TECHNICAL_FEASIBILITY]),
        "institutional_checkpoint_reuse_count": len(checkpoint_outputs[LensType.INSTITUTIONAL_REGIONAL]),
        "financial_checkpoint_reuse_count": len(checkpoint_outputs[LensType.FINANCIAL_ADOPTION]),
        "messages": [
            "**Stage 4**: final artifact absent; exact per-lens checkpoints loaded where available."
        ],
    }


async def _run_stage4_lens(
    state: AgentState,
    *,
    lens_type: LensType,
    state_field: str,
    progress_prefix: str,
) -> dict[str, Any]:
    if state.get("stage4_artifact_reused"):
        return {"messages": [f"**{lens_type.value}**: reused frozen Stage 4 assessment(s)."]}

    stage3_artifact = _load_exact_stage3_artifact(state)
    evidence_pack = state["evidence_pack"]
    decision_objects = [DecisionObject.model_validate(item) for item in state.get("decision_objects") or []]
    existing = {
        str(item["decision_object_id"]): item
        for item in (state.get(state_field) or [])
    }
    outputs: list[dict[str, Any]] = []
    for index, decision_object in enumerate(decision_objects, start=1):
        cached = existing.get(decision_object.decision_object_id)
        if cached is not None:
            parsed = validate_lens_assessment(
                cached,
                decision_object=decision_object,
                evidence_pack=evidence_pack,
                expected_lens_type=lens_type,
            )
            outputs.append(parsed.model_dump(mode="json"))
            continue

        payload = build_stage4_evaluation_payload(
            decision_object=decision_object,
            evidence_pack=evidence_pack,
        )
        structured_llm = get_llm().with_structured_output(
            LensAssessment,
            method="json_schema",
            progress_label=f"{progress_prefix}-{index}of{len(decision_objects)}",
            response_schema=build_constrained_lens_response_schema(
                decision_object=decision_object,
                evidence_pack=evidence_pack,
                expected_lens_type=lens_type,
            ),
            semantic_validator=lambda assessment, obj=decision_object: validate_lens_assessment(
                assessment,
                decision_object=obj,
                evidence_pack=evidence_pack,
                expected_lens_type=lens_type,
            ),
        )
        raw = await structured_llm.ainvoke(
            [
                SystemMessage(content=SYSTEM_PROMPT_BY_LENS[lens_type.value]),
                HumanMessage(
                    content=STAGE4_EVALUATION_USER_PROMPT.format(
                        evaluation_payload=json.dumps(
                            payload,
                            ensure_ascii=False,
                            indent=2,
                            sort_keys=True,
                        )
                    )
                ),
            ]
        )
        assessment = raw.model_dump(mode="json")
        checkpoint_path = freeze_stage4_lens_checkpoint(
            project_root=_project_root(),
            evidence_pack=evidence_pack,
            stage3_artifact=stage3_artifact,
            decision_object_id=decision_object.decision_object_id,
            lens_type=lens_type,
            assessment=assessment,
        )
        print(f"Stage 4 {lens_type.value} checkpoint saved immediately: {checkpoint_path}", flush=True)
        outputs.append(assessment)

    return {
        state_field: outputs,
        "messages": [
            f"**{lens_type.value}**: completed {len(outputs)} information-isolated DecisionObject assessment(s)."
        ],
    }


async def technical_feasibility_lens_node(state: AgentState) -> AgentState:
    return await _run_stage4_lens(
        state,
        lens_type=LensType.TECHNICAL_FEASIBILITY,
        state_field="technical_lens_assessments",
        progress_prefix="Stage4A-Technical",
    )


async def institutional_regional_lens_node(state: AgentState) -> AgentState:
    return await _run_stage4_lens(
        state,
        lens_type=LensType.INSTITUTIONAL_REGIONAL,
        state_field="institutional_lens_assessments",
        progress_prefix="Stage4B-InstitutionalRegional",
    )


async def financial_adoption_lens_node(state: AgentState) -> AgentState:
    return await _run_stage4_lens(
        state,
        lens_type=LensType.FINANCIAL_ADOPTION,
        state_field="financial_lens_assessments",
        progress_prefix="Stage4C-FinancialAdoption",
    )


def stage4_complete_node(state: AgentState) -> AgentState:
    if state.get("stage4_artifact_reused"):
        if not state.get("stage4_evaluation_bundle"):
            raise ValueError("Reused Stage 4 artifact is missing its evaluation bundle")
        return {
            "stage4_complete": True,
            "messages": [f"**Stage 4**: reused immutable artifact {state.get('stage4_artifact_path')}."],
        }

    stage3_artifact = _load_exact_stage3_artifact(state)
    evidence_pack = state["evidence_pack"]
    decision_objects = [DecisionObject.model_validate(item) for item in state.get("decision_objects") or []]
    lens_lists = {
        LensType.TECHNICAL_FEASIBILITY: state.get("technical_lens_assessments") or [],
        LensType.INSTITUTIONAL_REGIONAL: state.get("institutional_lens_assessments") or [],
        LensType.FINANCIAL_ADOPTION: state.get("financial_lens_assessments") or [],
    }
    maps: dict[LensType, dict[str, dict[str, Any]]] = {}
    for lens_type, assessments in lens_lists.items():
        maps[lens_type] = {
            str(item["decision_object_id"]): item
            for item in assessments
        }
        if len(maps[lens_type]) != len(decision_objects):
            raise ValueError(
                f"Stage 4 completion requires one {lens_type.value} assessment per DecisionObject"
            )

    evaluations = [
        DecisionLensEvaluation(
            decision_object_id=obj.decision_object_id,
            technical_feasibility=maps[LensType.TECHNICAL_FEASIBILITY][obj.decision_object_id],
            institutional_regional=maps[LensType.INSTITUTIONAL_REGIONAL][obj.decision_object_id],
            financial_adoption=maps[LensType.FINANCIAL_ADOPTION][obj.decision_object_id],
        )
        for obj in decision_objects
    ]
    bundle = validate_stage4_evaluation_bundle(
        Stage4EvaluationBundle(case_id=evidence_pack["case_id"], evaluations=evaluations),
        decision_objects=decision_objects,
        evidence_pack=evidence_pack,
    )
    artifact, artifact_path = freeze_stage4_artifact(
        project_root=_project_root(),
        evidence_pack=evidence_pack,
        stage3_artifact=stage3_artifact,
        evaluation_bundle=bundle,
    )
    clear_stage4_lens_checkpoints(
        project_root=_project_root(),
        evidence_pack=evidence_pack,
        stage3_artifact=stage3_artifact,
    )
    return {
        "stage4_complete": True,
        "stage4_artifact_reused": False,
        "stage4_input_fingerprint": artifact["input_fingerprint"],
        "stage4_artifact_path": str(artifact_path),
        "stage4_evaluation_bundle": artifact["evaluation_bundle"],
        "messages": [
            f"**Stage 4**: froze {len(evaluations)} three-lens evaluation unit(s) at {artifact_path}."
        ],
    }

