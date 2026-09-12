import os
from pathlib import Path
from langchain_core.messages import HumanMessage, SystemMessage
from state import AgentState
from Stage0.agent_input import build_agent_input
from Stage0.schema import EvaluationMode
from Stage1.artifact import freeze_stage1_artifact, load_frozen_stage1_artifact
from Stage1.schema import CriticAssessment, CriticType, validate_critic_assessment
from llm_runtime import LlamaCppChatClient, assert_llama_server_ready

from prompts import (
    ATTACK_FEASIBILITY_SYSTEM_PROMPT,
    DEFENSE_ROBUSTNESS_SYSTEM_PROMPT,
    STAGE1_EVALUATION_USER_PROMPT,
    MEDIATOR_SYSTEM_PROMPT,
    TECHNICAL_SYSTEM_PROMPT,
    REGIONAL_SYSTEM_PROMPT,
    FINANCE_BUSINESS_SYSTEM_PROMPT
)

def get_llm():
    assert_llama_server_ready()
    return LlamaCppChatClient()

def load_data_node(state: AgentState) -> AgentState:
    """Build the Agent input only from Stage 0 canonical forecast/evidence artifacts."""
    required_env = {
        "STAGE0_SNAPSHOT_ID": os.getenv("STAGE0_SNAPSHOT_ID"),
        "STAGE0_THREAT": os.getenv("STAGE0_THREAT"),
        "STAGE0_PMT": os.getenv("STAGE0_PMT"),
        "STAGE0_ANALYSIS_CUTOFF": os.getenv("STAGE0_ANALYSIS_CUTOFF"),
    }
    missing = [name for name, value in required_env.items() if not value]
    if missing:
        raise ValueError(f"Missing required Stage 0 runtime configuration: {', '.join(missing)}")

    project_root = Path(__file__).resolve().parents[1]
    snapshot_id = required_env["STAGE0_SNAPSHOT_ID"]
    threat = required_env["STAGE0_THREAT"]
    pmt = required_env["STAGE0_PMT"]
    analysis_cutoff = required_env["STAGE0_ANALYSIS_CUTOFF"]
    evaluation_mode = EvaluationMode(os.getenv("STAGE0_EVALUATION_MODE", EvaluationMode.CURRENT_REASSESSMENT.value))
    case_id = os.getenv("STAGE0_CASE_ID", f"{threat}__{pmt}")

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
    artifact, identity, artifact_path = load_frozen_stage1_artifact(
        project_root=project_root,
        evidence_pack=pack_dict,
        forecast_data=agent_input["forecast_data"],
    )
    artifact_reused = artifact is not None
    if artifact_reused:
        assessments = artifact["assessments"]
        attack_assessment = assessments["attack_feasibility"]
        defense_assessment = assessments["defense_robustness"]
        print(f"Stage 1 frozen artifact reused: {artifact_path}\n")
    else:
        attack_assessment = None
        defense_assessment = None
        print(f"Stage 1 artifact not found for fingerprint {identity['input_fingerprint']}; critics will run.\n")
    print(f"Stage 0 EvidencePack loaded: {case_id} ({len(pack_dict['evidence'])} evidence records)\n")
    return {
        "forecast_data": agent_input["forecast_data"],
        "evidence_pack": pack_dict,
        "attack_assessment": attack_assessment,
        "defense_assessment": defense_assessment,
        "stage1_complete": False,
        "stage1_artifact_reused": artifact_reused,
        "stage1_input_fingerprint": identity["input_fingerprint"],
        "stage1_artifact_path": str(artifact_path),
        "iteration_count": 0,
        "messages": []
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
    structured_llm = get_llm().with_structured_output(CriticAssessment, method="json_schema")
    raw_assessment = await structured_llm.ainvoke(
        [
            SystemMessage(content=prompt),
            HumanMessage(content=STAGE1_EVALUATION_USER_PROMPT),
        ]
    )
    assessment = validate_critic_assessment(
        raw_assessment,
        evidence_pack=evidence_pack,
        expected_critic_type=critic_type,
    )
    return assessment.model_dump(mode="json")


async def attack_feasibility_critic_node(state: AgentState) -> AgentState:
    """Stage 1A: independently evaluate the Threat side of the forecast."""
    if state.get("stage1_artifact_reused"):
        return {"messages": ["**Attack Feasibility Critic**: reused frozen Stage 1 assessment."]}
    print("--- Stage 1A: Attack Feasibility Critic ---\n")
    assessment = await _run_stage1_critic(
        state,
        critic_type=CriticType.ATTACK_FEASIBILITY,
        prompt_template=ATTACK_FEASIBILITY_SYSTEM_PROMPT,
    )
    return {
        "attack_assessment": assessment,
        "messages": [f"**Attack Feasibility Critic**: {assessment}"],
    }


async def defense_robustness_critic_node(state: AgentState) -> AgentState:
    """Stage 1B: independently evaluate the PMT side of the forecast."""
    if state.get("stage1_artifact_reused"):
        return {"messages": ["**Defense Robustness Critic**: reused frozen Stage 1 assessment."]}
    print("--- Stage 1B: Defense Robustness Critic ---\n")
    assessment = await _run_stage1_critic(
        state,
        critic_type=CriticType.DEFENSE_ROBUSTNESS,
        prompt_template=DEFENSE_ROBUSTNESS_SYSTEM_PROMPT,
    )
    return {
        "defense_assessment": assessment,
        "messages": [f"**Defense Robustness Critic**: {assessment}"],
    }


def stage1_complete_node(state: AgentState) -> AgentState:
    """Join the two independent branches without performing Stage 2 debate."""
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

    project_root = Path(__file__).resolve().parents[1]
    artifact, artifact_path = freeze_stage1_artifact(
        project_root=project_root,
        evidence_pack=evidence_pack,
        forecast_data=state["forecast_data"],
        attack_assessment=state["attack_assessment"],
        defense_assessment=state["defense_assessment"],
    )
    return {
        "stage1_complete": True,
        "stage1_input_fingerprint": artifact["input_fingerprint"],
        "stage1_artifact_path": str(artifact_path),
        "messages": [
            f"**Stage 1**: independent pre-assessments frozen at {artifact_path}; Stage 2 debate has not run."
        ],
    }

async def mediator_node(state: AgentState) -> AgentState:
    print("--- Mediator Turn ---\n")
    llm = get_llm()
    
    iteration_count = state["iteration_count"] + 1
    
    prompt = f"""
{MEDIATOR_SYSTEM_PROMPT.format(
    attack_plan=state["attack_plan"],
    defense_plan=state["defense_plan"],
    forecast_data=state["forecast_data"],
    iteration_count=iteration_count
)}

**Evidence boundary**: Use only the Stage 0 forecast/evidence payload above as factual support. Do not retrieve or invent additional evidence.
"""
    
    response = llm.invoke([SystemMessage(content=prompt)])

    return {
        "mediator_review": response.content,
        "iteration_count": iteration_count,
        "messages": [f"**Mediator (Turn {iteration_count})**: {response.content}"]
    }

async def technical_agent_node(state: AgentState) -> AgentState:
    """Legacy Technical Implementation Agent using only the frozen Stage 0 payload."""
    print("--- Technical Implementation Agent Turn ---\n")
    llm = get_llm()
    
    prompt = f"""
{TECHNICAL_SYSTEM_PROMPT.format(
    mediator_review=state["mediator_review"],
    forecast_data=state["forecast_data"]
)}

**Evidence boundary**: Use only the Stage 0 forecast/evidence payload above as factual support. Do not retrieve or invent additional evidence.
"""
    
    response = llm.invoke([SystemMessage(content=prompt)])
    
    return {
        "technical_analysis": response.content,
        "messages": [f"**Technical Implementation Agent**: {response.content}"]
    }

async def regional_agent_node(state: AgentState) -> AgentState:
    """Legacy Regional Agent using only the frozen Stage 0 payload."""
    print("--- Regional Agent Turn ---\n")
    llm = get_llm()
    
    prompt = f"""
{REGIONAL_SYSTEM_PROMPT.format(
    technical_analysis=state["technical_analysis"],
    forecast_data=state["forecast_data"]
)}

**Evidence boundary**: Use only the Stage 0 forecast/evidence payload above as factual support. Do not retrieve or invent additional evidence.
"""
    
    response = llm.invoke([SystemMessage(content=prompt)])
    
    return {
        "regional_strategy": response.content,
        "messages": [f"**Regional Agent**: {response.content}"]
    }

async def finance_business_agent_node(state: AgentState) -> AgentState:
    """Legacy Finance-Business Agent using only the frozen Stage 0 payload."""
    print("--- Finance-Business Agent Turn ---\n")
    llm = get_llm()
    history = "\n\n".join(state["messages"])
    
    prompt = f"""
{FINANCE_BUSINESS_SYSTEM_PROMPT.format(
    technical_analysis=state["technical_analysis"],
    regional_strategy=state["regional_strategy"],
    messages=history,
    forecast_data=state["forecast_data"]
)}

**Evidence boundary**: Use only the Stage 0 forecast/evidence payload above as factual support. Do not retrieve or invent additional evidence.
"""
    
    response = llm.invoke([SystemMessage(content=prompt)])
    
    return {
        "finance_business_plan": response.content,
        "messages": [f"**Finance-Business Agent (Final Report)**: {response.content}"]
    }
