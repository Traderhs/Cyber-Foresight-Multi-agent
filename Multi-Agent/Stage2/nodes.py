from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Callable

from langchain_core.messages import HumanMessage, SystemMessage

from Pipeline.state import AgentState
from Stage1.artifact import load_frozen_stage1_artifact
from Stage1.nodes import get_llm
from Stage1.schema import (
    CriticAssessment,
    CriticType,
)
from Stage2.artifact import (
    build_stage2_identity,
    clear_stage2_step_checkpoints,
    freeze_stage2_artifact,
    freeze_stage2_step_checkpoint,
    load_frozen_stage2_artifact,
    load_stage2_step_checkpoint,
)
from Stage2.prompts import (
    STAGE2_DEBATE_APPENDIX,
    STAGE2_DEBATE_USER_PROMPT,
    STAGE2_MEDIATOR_SYSTEM_PROMPT,
    STAGE2_MEDIATOR_USER_PROMPT,
    STAGE2_POST_ASSESSMENT_APPENDIX,
    STAGE2_POST_USER_PROMPT,
    stage2_role_context,
)
from Stage2.schema import (
    CriticDebateTurn,
    DebateRound,
    MediatorSummary,
    Stage2DebateResult,
    build_constrained_debate_turn_schema,
    build_constrained_mediator_schema,
    build_constrained_post_assessment_schema,
    flatten_debate_exchanges,
    round_requires_followup,
    surfaced_evidence_ids,
    validate_debate_turn,
    validate_mediator_summary,
    validate_post_assessment,
    validate_stage2_result,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _forecast_data_visible_to_stage2_step(
    forecast_data: str,
    *,
    allowed_evidence_ids: list[str] | None,
) -> str:
    """Return the prompt-facing Stage 0 view, optionally hiding unsurfaced evidence records."""
    if allowed_evidence_ids is None:
        return forecast_data
    try:
        payload = json.loads(forecast_data)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Stage 2 requires forecast_data to be the canonical JSON prompt payload") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("evidence"), list):
        raise ValueError("Stage 2 forecast_data JSON is missing the evidence list")
    allowed = set(allowed_evidence_ids)
    available = {
        str(record.get("evidence_id"))
        for record in payload["evidence"]
        if isinstance(record, dict) and record.get("evidence_id")
    }
    unknown = sorted(allowed - available)
    if unknown:
        raise ValueError(f"Stage 2 visibility filter references evidence absent from forecast_data: {unknown}")
    payload["evidence"] = [
        record
        for record in payload["evidence"]
        if isinstance(record, dict) and str(record.get("evidence_id")) in allowed
    ]
    payload["stage2_evidence_visibility"] = {
        "mode": "surfaced_only",
        "visible_evidence_count": len(payload["evidence"]),
        "visible_evidence_ids": sorted(allowed),
        "note": (
            "This Stage 2 step intentionally hides frozen EvidencePack records that were not surfaced before this "
            "step; hidden records must not influence the current Stage 2 judgment."
        ),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def _load_exact_stage1_artifact(state: AgentState) -> dict[str, Any]:
    artifact, _, path = load_frozen_stage1_artifact(
        project_root=_project_root(),
        evidence_pack=state["evidence_pack"],
        forecast_data=state["forecast_data"],
    )
    if artifact is None:
        raise ValueError(
            "Stage 2 requires the exact frozen Stage 1 artifact before debate; "
            f"expected artifact at {path}"
        )
    return artifact


def _assert_state_pre_assessments_match_stage1_artifact(
    state: AgentState,
    stage1_artifact: dict[str, Any],
) -> None:
    assessments = stage1_artifact.get("assessments") or {}
    expected_attack = assessments.get("attack_feasibility")
    expected_defense = assessments.get("defense_robustness")
    if state.get("attack_assessment") != expected_attack:
        raise ValueError("Stage 2 state Attack pre-assessment does not match the exact frozen Stage 1 artifact")
    if state.get("defense_assessment") != expected_defense:
        raise ValueError("Stage 2 state Defense pre-assessment does not match the exact frozen Stage 1 artifact")


def prepare_stage2_node(state: AgentState) -> AgentState:
    """Bind Stage 2 to the exact frozen Stage 1 output and reuse a final debate artifact when present."""
    if not state.get("stage1_complete"):
        raise ValueError("Stage 2 preparation requires completed Stage 1")
    if not state.get("attack_assessment") or not state.get("defense_assessment"):
        raise ValueError("Stage 2 requires both frozen Stage 1 pre-assessments")
    if not state.get("evidence_pack") or not state.get("forecast_data"):
        raise ValueError("Stage 2 requires the complete Stage 0 input")

    stage1_artifact = _load_exact_stage1_artifact(state)
    _assert_state_pre_assessments_match_stage1_artifact(state, stage1_artifact)
    artifact, identity, artifact_path = load_frozen_stage2_artifact(
        project_root=_project_root(),
        evidence_pack=state["evidence_pack"],
        forecast_data=state["forecast_data"],
        stage1_artifact=stage1_artifact,
        attack_pre_assessment=state["attack_assessment"],
        defense_pre_assessment=state["defense_assessment"],
    )
    if artifact is not None:
        result = artifact["debate_result"]
        print(f"Stage 2 frozen artifact reused: {artifact_path}\n", flush=True)
        return {
            "stage2_artifact_reused": True,
            "stage2_complete": False,
            "stage2_input_fingerprint": identity["input_fingerprint"],
            "stage2_artifact_path": str(artifact_path),
            "stage2_debate_result": result,
            "debate_rounds": result["rounds"],
            "attack_post_assessment": result["attack_post_assessment"],
            "defense_post_assessment": result["defense_post_assessment"],
        }

    print(
        f"Stage 2 artifact not found for fingerprint {identity['input_fingerprint']}; "
        "structured debate will run/resume from exact step checkpoints.\n",
        flush=True,
    )
    return {
        "stage2_artifact_reused": False,
        "stage2_complete": False,
        "stage2_input_fingerprint": identity["input_fingerprint"],
        "stage2_artifact_path": str(artifact_path),
        "stage2_debate_result": None,
        "debate_rounds": [],
        "attack_post_assessment": None,
        "defense_post_assessment": None,
    }


async def _load_or_generate_turn(
    *,
    state: AgentState,
    llm_provider: Callable[[], Any],
    identity: dict[str, Any],
    responder: CriticType,
    round_index: int,
    previous_rounds: list[dict[str, Any]],
) -> CriticDebateTurn:
    step_name = f"round{round_index}_{'attack' if responder == CriticType.ATTACK_FEASIBILITY else 'defense'}"
    own_pre = (
        state["attack_assessment"]
        if responder == CriticType.ATTACK_FEASIBILITY
        else state["defense_assessment"]
    )
    opponent_pre = (
        state["defense_assessment"]
        if responder == CriticType.ATTACK_FEASIBILITY
        else state["attack_assessment"]
    )
    allowed_evidence_ids = None
    if round_index > 1:
        allowed_evidence_ids = surfaced_evidence_ids(
            attack_assessment=state["attack_assessment"],
            defense_assessment=state["defense_assessment"],
            previous_rounds=previous_rounds,
        )
    response_schema = build_constrained_debate_turn_schema(
        evidence_pack=state["evidence_pack"],
        opponent_assessment=opponent_pre,
        expected_responder=responder,
        round_index=round_index,
        allowed_evidence_ids=allowed_evidence_ids,
    )
    dependency_payload = {
        "step_kind": "critic_debate_turn",
        "responder": responder.value,
        "round_index": round_index,
        "own_pre_assessment": own_pre,
        "opponent_pre_assessment": opponent_pre,
        "previous_rounds": previous_rounds,
        "response_schema": response_schema,
    }
    checkpoint, checkpoint_path = load_stage2_step_checkpoint(
        project_root=_project_root(),
        identity=identity,
        step_name=step_name,
        dependency_payload=dependency_payload,
    )
    if checkpoint is not None:
        parsed = validate_debate_turn(
            checkpoint["payload"],
            evidence_pack=state["evidence_pack"],
            opponent_assessment=opponent_pre,
            expected_responder=responder,
            round_index=round_index,
            allowed_evidence_ids=allowed_evidence_ids,
        )
        print(f"Stage 2 exact step checkpoint reused: {checkpoint_path}", flush=True)
        return parsed

    llm = llm_provider()
    prompt_forecast_data = _forecast_data_visible_to_stage2_step(
        state["forecast_data"],
        allowed_evidence_ids=allowed_evidence_ids,
    )
    base_prompt = stage2_role_context(
        critic_type=responder.value,
        forecast_data=prompt_forecast_data,
    )
    system_prompt = base_prompt + STAGE2_DEBATE_APPENDIX.format(
        own_pre_assessment=_json(own_pre),
        opponent_pre_assessment=_json(opponent_pre),
        previous_rounds=_json(previous_rounds),
        round_index=round_index,
    )
    progress_label = (
        f"Stage2-R{round_index}-AttackResponse"
        if responder == CriticType.ATTACK_FEASIBILITY
        else f"Stage2-R{round_index}-DefenseResponse"
    )
    runner = llm.with_structured_output(
        CriticDebateTurn,
        method="json_schema",
        progress_label=progress_label,
        response_schema=response_schema,
        semantic_validator=lambda turn: validate_debate_turn(
            turn,
            evidence_pack=state["evidence_pack"],
            opponent_assessment=opponent_pre,
            expected_responder=responder,
            round_index=round_index,
            allowed_evidence_ids=allowed_evidence_ids,
        ),
    )
    parsed = await runner.ainvoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=STAGE2_DEBATE_USER_PROMPT),
        ]
    )
    freeze_stage2_step_checkpoint(
        project_root=_project_root(),
        identity=identity,
        step_name=step_name,
        payload=parsed.model_dump(mode="json"),
        dependency_payload=dependency_payload,
    )
    return parsed


async def _load_or_generate_mediator(
    *,
    state: AgentState,
    llm_provider: Callable[[], Any],
    identity: dict[str, Any],
    round_index: int,
    attack_turn: CriticDebateTurn,
    defense_turn: CriticDebateTurn,
    previous_rounds: list[dict[str, Any]],
) -> MediatorSummary:
    step_name = f"round{round_index}_mediator"
    mediator_allowed_evidence_ids = surfaced_evidence_ids(
        attack_assessment=state["attack_assessment"],
        defense_assessment=state["defense_assessment"],
        attack_turn=attack_turn,
        defense_turn=defense_turn,
        previous_rounds=previous_rounds,
    )
    mediator_evidence_context = _forecast_data_visible_to_stage2_step(
        state["forecast_data"],
        allowed_evidence_ids=mediator_allowed_evidence_ids,
    )
    response_schema = build_constrained_mediator_schema(
        evidence_pack=state["evidence_pack"],
        attack_assessment=state["attack_assessment"],
        defense_assessment=state["defense_assessment"],
        round_index=round_index,
        attack_turn=attack_turn,
        defense_turn=defense_turn,
        previous_rounds=previous_rounds,
    )
    dependency_payload = {
        "step_kind": "mediator_summary",
        "round_index": round_index,
        "attack_pre_assessment": state["attack_assessment"],
        "defense_pre_assessment": state["defense_assessment"],
        "attack_turn": attack_turn.model_dump(mode="json"),
        "defense_turn": defense_turn.model_dump(mode="json"),
        "previous_rounds": previous_rounds,
        "mediator_visible_evidence_ids": mediator_allowed_evidence_ids,
        "mediator_evidence_context": mediator_evidence_context,
        "response_schema": response_schema,
    }
    checkpoint, checkpoint_path = load_stage2_step_checkpoint(
        project_root=_project_root(),
        identity=identity,
        step_name=step_name,
        dependency_payload=dependency_payload,
    )
    if checkpoint is not None:
        parsed = validate_mediator_summary(
            checkpoint["payload"],
            evidence_pack=state["evidence_pack"],
            attack_assessment=state["attack_assessment"],
            defense_assessment=state["defense_assessment"],
            attack_turn=attack_turn,
            defense_turn=defense_turn,
            round_index=round_index,
            previous_rounds=previous_rounds,
        )
        print(f"Stage 2 exact step checkpoint reused: {checkpoint_path}", flush=True)
        return parsed

    llm = llm_provider()
    system_prompt = STAGE2_MEDIATOR_SYSTEM_PROMPT.format(
        attack_pre_assessment=_json(state["attack_assessment"]),
        defense_pre_assessment=_json(state["defense_assessment"]),
        attack_turn=_json(attack_turn),
        defense_turn=_json(defense_turn),
        previous_rounds=_json(previous_rounds),
        mediator_evidence_context=mediator_evidence_context,
        round_index=round_index,
    )
    runner = llm.with_structured_output(
        MediatorSummary,
        method="json_schema",
        progress_label=f"Stage2-R{round_index}-Mediator",
        response_schema=response_schema,
        semantic_validator=lambda summary: validate_mediator_summary(
            summary,
            evidence_pack=state["evidence_pack"],
            attack_assessment=state["attack_assessment"],
            defense_assessment=state["defense_assessment"],
            attack_turn=attack_turn,
            defense_turn=defense_turn,
            round_index=round_index,
            previous_rounds=previous_rounds,
        ),
    )
    parsed = await runner.ainvoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=STAGE2_MEDIATOR_USER_PROMPT),
        ]
    )
    freeze_stage2_step_checkpoint(
        project_root=_project_root(),
        identity=identity,
        step_name=step_name,
        payload=parsed.model_dump(mode="json"),
        dependency_payload=dependency_payload,
    )
    return parsed


async def _load_or_generate_post_assessment(
    *,
    state: AgentState,
    llm_provider: Callable[[], Any],
    identity: dict[str, Any],
    critic_type: CriticType,
    rounds: list[dict[str, Any]],
) -> CriticAssessment:
    role_name = "attack" if critic_type == CriticType.ATTACK_FEASIBILITY else "defense"
    step_name = f"post_{role_name}"
    own_pre = (
        state["attack_assessment"]
        if critic_type == CriticType.ATTACK_FEASIBILITY
        else state["defense_assessment"]
    )
    opponent_pre = (
        state["defense_assessment"]
        if critic_type == CriticType.ATTACK_FEASIBILITY
        else state["attack_assessment"]
    )
    dependency_payload = {
        "step_kind": "post_assessment",
        "critic_type": critic_type.value,
        "own_pre_assessment": own_pre,
        "opponent_pre_assessment": opponent_pre,
        "rounds": rounds,
    }
    response_schema = build_constrained_post_assessment_schema(
        evidence_pack=state["evidence_pack"],
        expected_critic_type=critic_type,
        attack_pre_assessment=state["attack_assessment"],
        defense_pre_assessment=state["defense_assessment"],
        rounds=rounds,
    )
    post_allowed_evidence_ids = surfaced_evidence_ids(
        attack_assessment=state["attack_assessment"],
        defense_assessment=state["defense_assessment"],
        previous_rounds=rounds,
    )
    dependency_payload["response_schema"] = response_schema
    checkpoint, checkpoint_path = load_stage2_step_checkpoint(
        project_root=_project_root(),
        identity=identity,
        step_name=step_name,
        dependency_payload=dependency_payload,
    )
    if checkpoint is not None:
        parsed = validate_post_assessment(
            checkpoint["payload"],
            evidence_pack=state["evidence_pack"],
            expected_critic_type=critic_type,
            attack_pre_assessment=state["attack_assessment"],
            defense_pre_assessment=state["defense_assessment"],
            rounds=rounds,
        )
        print(f"Stage 2 exact step checkpoint reused: {checkpoint_path}", flush=True)
        return parsed

    llm = llm_provider()
    prompt_forecast_data = _forecast_data_visible_to_stage2_step(
        state["forecast_data"],
        allowed_evidence_ids=post_allowed_evidence_ids,
    )
    base_prompt = stage2_role_context(
        critic_type=critic_type.value,
        forecast_data=prompt_forecast_data,
    )
    system_prompt = base_prompt + STAGE2_POST_ASSESSMENT_APPENDIX.format(
        own_pre_assessment=_json(own_pre),
        opponent_pre_assessment=_json(opponent_pre),
        debate_rounds=_json(rounds),
    )
    runner = llm.with_structured_output(
        CriticAssessment,
        method="json_schema",
        progress_label=(
            "Stage2-Post-AttackAssessment"
            if critic_type == CriticType.ATTACK_FEASIBILITY
            else "Stage2-Post-DefenseAssessment"
        ),
        response_schema=response_schema,
        semantic_validator=lambda assessment: validate_post_assessment(
            assessment,
            evidence_pack=state["evidence_pack"],
            expected_critic_type=critic_type,
            attack_pre_assessment=state["attack_assessment"],
            defense_pre_assessment=state["defense_assessment"],
            rounds=rounds,
        ),
    )
    parsed = await runner.ainvoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=STAGE2_POST_USER_PROMPT),
        ]
    )
    freeze_stage2_step_checkpoint(
        project_root=_project_root(),
        identity=identity,
        step_name=step_name,
        payload=parsed.model_dump(mode="json"),
        dependency_payload=dependency_payload,
    )
    return parsed


async def stage2_debate_node(state: AgentState) -> AgentState:
    """Run at most two bounded debate rounds, then independent post-assessments."""
    if state.get("stage2_artifact_reused"):
        return {"messages": ["**Stage 2**: reused frozen structured-debate result."]}

    print("--- Stage 2: Evidence-grounded Structured Debate ---\n", flush=True)
    stage1_artifact = _load_exact_stage1_artifact(state)
    _assert_state_pre_assessments_match_stage1_artifact(state, stage1_artifact)
    identity = build_stage2_identity(
        evidence_pack=state["evidence_pack"],
        forecast_data=state["forecast_data"],
        stage1_artifact=stage1_artifact,
    )
    llm_client: Any | None = None

    def llm_provider() -> Any:
        nonlocal llm_client
        if llm_client is None:
            llm_client = get_llm()
        return llm_client

    rounds: list[DebateRound] = []

    for round_index in (1, 2):
        previous_rounds = [item.model_dump(mode="json") for item in rounds]
        # Attack and Defense turns are information-isolated within the same
        # round: neither current turn is included in the other's prompt. Run
        # them concurrently on the two-slot llama.cpp profile, then wait for
        # both before invoking the Mediator.
        attack_turn, defense_turn = await asyncio.gather(
            _load_or_generate_turn(
                state=state,
                llm_provider=llm_provider,
                identity=identity,
                responder=CriticType.ATTACK_FEASIBILITY,
                round_index=round_index,
                previous_rounds=previous_rounds,
            ),
            _load_or_generate_turn(
                state=state,
                llm_provider=llm_provider,
                identity=identity,
                responder=CriticType.DEFENSE_ROBUSTNESS,
                round_index=round_index,
                previous_rounds=previous_rounds,
            ),
        )
        mediator = await _load_or_generate_mediator(
            state=state,
            llm_provider=llm_provider,
            identity=identity,
            round_index=round_index,
            attack_turn=attack_turn,
            defense_turn=defense_turn,
            previous_rounds=previous_rounds,
        )
        round_result = DebateRound(
            round_index=round_index,
            attack_turn=attack_turn,
            defense_turn=defense_turn,
            mediator_summary=mediator,
        )
        rounds.append(round_result)
        if round_index == 1 and not round_requires_followup(round_result):
            print("Stage 2 debate ended after round 1: Mediator adjudication selected STOP.", flush=True)
            break

    serialized_rounds = [item.model_dump(mode="json") for item in rounds]
    attack_post, defense_post = await asyncio.gather(
        _load_or_generate_post_assessment(
            state=state,
            llm_provider=llm_provider,
            identity=identity,
            critic_type=CriticType.ATTACK_FEASIBILITY,
            rounds=serialized_rounds,
        ),
        _load_or_generate_post_assessment(
            state=state,
            llm_provider=llm_provider,
            identity=identity,
            critic_type=CriticType.DEFENSE_ROBUSTNESS,
            rounds=serialized_rounds,
        ),
    )
    final_mediator = rounds[-1].mediator_summary
    result = Stage2DebateResult(
        case_id=state["evidence_pack"]["case_id"],
        attack_pre_assessment=state["attack_assessment"],
        defense_pre_assessment=state["defense_assessment"],
        rounds=rounds,
        exchanges=flatten_debate_exchanges(rounds),
        resolved_claims=[
            item
            for item in final_mediator.claim_matches
            if item.status.value == "RESOLVED"
        ],
        unresolved_claims=[
            item
            for item in final_mediator.claim_matches
            if item.status.value == "UNRESOLVED"
        ],
        evidence_conflicts=final_mediator.evidence_conflicts,
        evidence_gaps=final_mediator.evidence_gaps,
        final_adjudications=final_mediator.adjudications,
        attack_post_assessment=attack_post,
        defense_post_assessment=defense_post,
        stance_changes={
            "attack_feasibility": attack_post.stance - int(state["attack_assessment"]["stance"]),
            "defense_robustness": defense_post.stance - int(state["defense_assessment"]["stance"]),
        },
    )
    parsed = validate_stage2_result(
        result,
        evidence_pack=state["evidence_pack"],
        attack_pre_assessment=state["attack_assessment"],
        defense_pre_assessment=state["defense_assessment"],
    )
    return {
        "stage2_debate_result": parsed.model_dump(mode="json"),
        "debate_rounds": serialized_rounds,
        "attack_post_assessment": attack_post.model_dump(mode="json"),
        "defense_post_assessment": defense_post.model_dump(mode="json"),
        "messages": [
            f"**Stage 2**: completed {len(rounds)} debate round(s); "
            f"stance changes attack={result.stance_changes.attack_feasibility}, "
            f"defense={result.stance_changes.defense_robustness}."
        ],
    }


def stage2_complete_node(state: AgentState) -> AgentState:
    """Validate/freeze Stage 2 and remove redundant exact step checkpoints."""
    if not state.get("stage2_debate_result"):
        raise ValueError("Stage 2 completion requires a structured debate result")
    if state.get("stage2_artifact_reused"):
        return {
            "stage2_complete": True,
            "messages": [f"**Stage 2**: reused immutable artifact {state.get('stage2_artifact_path')}."],
        }

    stage1_artifact = _load_exact_stage1_artifact(state)
    _assert_state_pre_assessments_match_stage1_artifact(state, stage1_artifact)
    artifact, artifact_path = freeze_stage2_artifact(
        project_root=_project_root(),
        evidence_pack=state["evidence_pack"],
        forecast_data=state["forecast_data"],
        stage1_artifact=stage1_artifact,
        attack_pre_assessment=state["attack_assessment"],
        defense_pre_assessment=state["defense_assessment"],
        debate_result=state["stage2_debate_result"],
    )
    identity = build_stage2_identity(
        evidence_pack=state["evidence_pack"],
        forecast_data=state["forecast_data"],
        stage1_artifact=stage1_artifact,
    )
    clear_stage2_step_checkpoints(project_root=_project_root(), identity=identity)
    return {
        "stage2_complete": True,
        "stage2_input_fingerprint": artifact["input_fingerprint"],
        "stage2_artifact_path": str(artifact_path),
        "messages": [f"**Stage 2**: structured debate frozen at {artifact_path}."],
    }
