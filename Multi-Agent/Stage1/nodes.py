from __future__ import annotations

from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from Pipeline.state import AgentState
from Stage1.artifact import (
    clear_stage1_critic_checkpoints,
    freeze_stage1_artifact,
    freeze_stage1_critic_checkpoint,
    load_frozen_stage1_artifact,
    load_stage1_critic_checkpoint,
)
from Stage1.prompts import (
    ATTACK_FEASIBILITY_SYSTEM_PROMPT,
    DEFENSE_ROBUSTNESS_SYSTEM_PROMPT,
    STAGE1_EVALUATION_USER_PROMPT,
)
from Stage1.runtime import LlamaCppChatClient, assert_llama_server_ready
from Stage1.schema import (
    CriticAssessment,
    CriticType,
    build_constrained_critic_response_schema,
    validate_critic_assessment,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_llm() -> LlamaCppChatClient:
    assert_llama_server_ready()
    return LlamaCppChatClient(request_concurrency=2)


def prepare_stage1_node(state: AgentState) -> AgentState:
    """Resolve exact Stage 1 artifact identity and reuse it when already frozen."""
    evidence_pack = state.get("evidence_pack")
    forecast_data = state.get("forecast_data")
    if not evidence_pack or not forecast_data:
        raise ValueError("Stage 1 preparation requires the complete Stage 0 input")

    artifact, identity, artifact_path = load_frozen_stage1_artifact(
        project_root=_project_root(),
        evidence_pack=evidence_pack,
        forecast_data=forecast_data,
    )
    reused = artifact is not None
    if reused:
        assessments = artifact["assessments"]
        attack_assessment = assessments["attack_feasibility"]
        defense_assessment = assessments["defense_robustness"]
        attack_checkpoint_reused = False
        defense_checkpoint_reused = False
        print(f"Stage 1 frozen artifact reused: {artifact_path}\n")
    else:
        attack_checkpoint, attack_checkpoint_path = load_stage1_critic_checkpoint(
            project_root=_project_root(),
            evidence_pack=evidence_pack,
            forecast_data=forecast_data,
            critic_type=CriticType.ATTACK_FEASIBILITY,
        )
        defense_checkpoint, defense_checkpoint_path = load_stage1_critic_checkpoint(
            project_root=_project_root(),
            evidence_pack=evidence_pack,
            forecast_data=forecast_data,
            critic_type=CriticType.DEFENSE_ROBUSTNESS,
        )
        attack_checkpoint_reused = attack_checkpoint is not None
        defense_checkpoint_reused = defense_checkpoint is not None
        attack_assessment = attack_checkpoint["assessment"] if attack_checkpoint else None
        defense_assessment = defense_checkpoint["assessment"] if defense_checkpoint else None
        print(
            f"Stage 1 artifact not found for fingerprint {identity['input_fingerprint']}; "
            "critics will run.\n"
        )
        if attack_checkpoint_reused:
            print(f"Stage 1A exact critic checkpoint reused: {attack_checkpoint_path}", flush=True)
        if defense_checkpoint_reused:
            print(f"Stage 1B exact critic checkpoint reused: {defense_checkpoint_path}", flush=True)

    return {
        "attack_assessment": attack_assessment,
        "defense_assessment": defense_assessment,
        "stage1_complete": False,
        "stage1_artifact_reused": reused,
        "attack_checkpoint_reused": attack_checkpoint_reused,
        "defense_checkpoint_reused": defense_checkpoint_reused,
        "stage1_input_fingerprint": identity["input_fingerprint"],
        "stage1_artifact_path": str(artifact_path),
    }


async def _run_stage1_critic(
    state: AgentState,
    *,
    critic_type: CriticType,
    prompt_template: str,
) -> dict[str, object]:
    evidence_pack = state.get("evidence_pack")
    if not evidence_pack:
        raise ValueError("Stage 1 requires a Stage 0 EvidencePack")

    prompt = prompt_template.format(forecast_data=state["forecast_data"])
    progress_label = (
        "Stage1A-AttackFeasibility"
        if critic_type == CriticType.ATTACK_FEASIBILITY
        else "Stage1B-DefenseRobustness"
    )
    structured_llm = get_llm().with_structured_output(
        CriticAssessment,
        method="json_schema",
        progress_label=progress_label,
        response_schema=build_constrained_critic_response_schema(
            evidence_pack=evidence_pack,
            expected_critic_type=critic_type,
        ),
        semantic_validator=lambda assessment: validate_critic_assessment(
            assessment,
            evidence_pack=evidence_pack,
            expected_critic_type=critic_type,
        ),
    )
    raw_assessment = await structured_llm.ainvoke(
        [
            SystemMessage(content=prompt),
            HumanMessage(content=STAGE1_EVALUATION_USER_PROMPT),
        ]
    )
    return raw_assessment.model_dump(mode="json")


async def attack_feasibility_critic_node(state: AgentState) -> AgentState:
    """Stage 1A: independently evaluate the Threat side of the forecast."""
    if state.get("stage1_artifact_reused"):
        return {"messages": ["**Attack Feasibility Critic**: reused frozen Stage 1 assessment."]}
    if state.get("attack_checkpoint_reused"):
        return {"messages": ["**Attack Feasibility Critic**: resumed from exact critic checkpoint."]}
    print("--- Stage 1A: Attack Feasibility Critic ---\n")
    assessment = await _run_stage1_critic(
        state,
        critic_type=CriticType.ATTACK_FEASIBILITY,
        prompt_template=ATTACK_FEASIBILITY_SYSTEM_PROMPT,
    )
    checkpoint_path = freeze_stage1_critic_checkpoint(
        project_root=_project_root(),
        evidence_pack=state["evidence_pack"],
        forecast_data=state["forecast_data"],
        critic_type=CriticType.ATTACK_FEASIBILITY,
        assessment=assessment,
    )
    print(f"Stage 1A checkpoint saved immediately: {checkpoint_path}", flush=True)
    return {
        "attack_assessment": assessment,
        "messages": [f"**Attack Feasibility Critic**: {assessment}"],
    }


async def defense_robustness_critic_node(state: AgentState) -> AgentState:
    """Stage 1B: independently evaluate the PMT side of the forecast."""
    if state.get("stage1_artifact_reused"):
        return {"messages": ["**Defense Robustness Critic**: reused frozen Stage 1 assessment."]}
    if state.get("defense_checkpoint_reused"):
        return {"messages": ["**Defense Robustness Critic**: resumed from exact critic checkpoint."]}
    print("--- Stage 1B: Defense Robustness Critic ---\n")
    assessment = await _run_stage1_critic(
        state,
        critic_type=CriticType.DEFENSE_ROBUSTNESS,
        prompt_template=DEFENSE_ROBUSTNESS_SYSTEM_PROMPT,
    )
    checkpoint_path = freeze_stage1_critic_checkpoint(
        project_root=_project_root(),
        evidence_pack=state["evidence_pack"],
        forecast_data=state["forecast_data"],
        critic_type=CriticType.DEFENSE_ROBUSTNESS,
        assessment=assessment,
    )
    print(f"Stage 1B checkpoint saved immediately: {checkpoint_path}", flush=True)
    return {
        "defense_assessment": assessment,
        "messages": [f"**Defense Robustness Critic**: {assessment}"],
    }


def stage1_complete_node(state: AgentState) -> AgentState:
    """Validate and freeze the independent pre-assessments; never run Stage 2 here."""
    evidence_pack = state.get("evidence_pack")
    if not evidence_pack:
        raise ValueError("Stage 1 completion requires a Stage 0 EvidencePack")
    if not state.get("attack_assessment") or not state.get("defense_assessment"):
        raise ValueError("Stage 1 completion requires both independent CriticAssessment outputs")

    validate_critic_assessment(
        state["attack_assessment"],
        evidence_pack=evidence_pack,
        expected_critic_type=CriticType.ATTACK_FEASIBILITY,
    )
    validate_critic_assessment(
        state["defense_assessment"],
        evidence_pack=evidence_pack,
        expected_critic_type=CriticType.DEFENSE_ROBUSTNESS,
    )
    if state.get("stage1_artifact_reused"):
        return {
            "stage1_complete": True,
            "messages": [
                f"**Stage 1**: reused immutable pre-assessments from {state.get('stage1_artifact_path')}."
            ],
        }

    artifact, artifact_path = freeze_stage1_artifact(
        project_root=_project_root(),
        evidence_pack=evidence_pack,
        forecast_data=state["forecast_data"],
        attack_assessment=state["attack_assessment"],
        defense_assessment=state["defense_assessment"],
    )
    clear_stage1_critic_checkpoints(
        project_root=_project_root(),
        evidence_pack=evidence_pack,
        forecast_data=state["forecast_data"],
    )
    return {
        "stage1_complete": True,
        "stage1_input_fingerprint": artifact["input_fingerprint"],
        "stage1_artifact_path": str(artifact_path),
        "messages": [
            f"**Stage 1**: independent pre-assessments frozen at {artifact_path}; "
            "Stage 2 debate has not run."
        ],
    }
