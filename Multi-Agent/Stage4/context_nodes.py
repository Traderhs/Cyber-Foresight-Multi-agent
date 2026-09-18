from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from Pipeline.state import AgentState
from Stage3.context_artifact import load_frozen_stage3_context_artifact
from Stage3.context_nodes import _load_exact_base_stage3_artifact
from Stage3.context_schema import ContextualDecisionObject
from Stage4.context_artifact import (
    clear_stage4_context_lens_checkpoints,
    freeze_stage4_context_artifact,
    freeze_stage4_context_lens_checkpoint,
    load_frozen_stage4_context_artifact,
    load_stage4_context_lens_checkpoint,
)
from Stage4.context_input import build_context_stage4_evaluation_payload
from Stage4.context_prompts import CONTEXT_STAGE4_EVALUATION_USER_PROMPT, CONTEXT_SYSTEM_PROMPT_BY_LENS
from Stage4.context_schema import (
    ContextLensAssessment,
    DecisionLensEvaluation,
    LensAssessment,
    LensType,
    Stage4EvaluationBundle,
    build_constrained_context_lens_response_schema,
    context_lens_equivalence_key,
    validate_context_lens_assessment,
    validate_context_stage4_bundle,
)
from Stage4.context_runtime import STAGE4_CLIENT_CONCURRENCY, get_context_llm


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_exact_stage3_context_artifact(state: AgentState) -> dict[str, Any]:
    if not state.get("stage3_context_complete"):
        raise ValueError("Contextual Stage 4 requires contextual Stage 3 completion")
    if not state.get("stage3_contextual_decision_bundle") or not state.get("contextual_decision_objects"):
        raise ValueError("Contextual Stage 4 requires the frozen contextual Stage 3 bundle")
    base_artifact = _load_exact_base_stage3_artifact(state)
    artifact, _, path = load_frozen_stage3_context_artifact(
        project_root=_project_root(),
        evidence_pack=state["evidence_pack"],
        base_stage3_artifact=base_artifact,
    )
    if artifact is None:
        raise ValueError("Contextual Stage 4 requires the exact contextual Stage 3 artifact")
    state_path = state.get("stage3_context_artifact_path")
    if not state_path or Path(state_path).resolve() != path.resolve():
        raise ValueError("Contextual Stage 4 state path does not match exact contextual Stage 3 artifact")
    if artifact.get("contextual_decision_bundle") != state.get("stage3_contextual_decision_bundle"):
        raise ValueError("Contextual Stage 4 state bundle drifted from exact contextual Stage 3 artifact")
    if artifact.get("contextual_decision_bundle", {}).get("contextual_decision_objects") != state.get(
        "contextual_decision_objects"
    ):
        raise ValueError("Contextual Stage 4 decision objects drifted from exact contextual Stage 3 artifact")
    return artifact


def _assessments_from_bundle(bundle: dict[str, Any], lens_type: LensType) -> list[dict[str, Any]]:
    key = {
        LensType.TECHNICAL_FEASIBILITY: "technical_feasibility",
        LensType.INSTITUTIONAL_REGIONAL: "institutional_regional",
        LensType.FINANCIAL_ADOPTION: "financial_adoption",
    }[lens_type]
    return [item[key] for item in bundle.get("evaluations", [])]


def prepare_stage4_context_node(state: AgentState) -> AgentState:
    stage3_artifact = _load_exact_stage3_context_artifact(state)
    frozen, identity, artifact_path = load_frozen_stage4_context_artifact(
        project_root=_project_root(),
        stage3_context_artifact=stage3_artifact,
    )
    if frozen is not None:
        bundle = frozen["evaluation_bundle"]
        return {
            "stage4_context_complete": False,
            "stage4_context_artifact_reused": True,
            "stage4_context_input_fingerprint": identity["input_fingerprint"],
            "stage4_context_artifact_path": str(artifact_path),
            "stage4_context_evaluation_bundle": bundle,
            "contextual_technical_lens_assessments": _assessments_from_bundle(
                bundle, LensType.TECHNICAL_FEASIBILITY
            ),
            "contextual_institutional_lens_assessments": _assessments_from_bundle(
                bundle, LensType.INSTITUTIONAL_REGIONAL
            ),
            "contextual_financial_lens_assessments": _assessments_from_bundle(
                bundle, LensType.FINANCIAL_ADOPTION
            ),
            "contextual_technical_checkpoint_reuse_count": 0,
            "contextual_institutional_checkpoint_reuse_count": 0,
            "contextual_financial_checkpoint_reuse_count": 0,
            "messages": [f"**Stage 4 Context**: reused frozen contextual lens artifact {artifact_path}."],
        }

    objects = [ContextualDecisionObject.model_validate(item) for item in state["contextual_decision_objects"]]
    checkpoint_outputs: dict[LensType, list[dict[str, Any]]] = {lens: [] for lens in LensType}
    for obj in objects:
        for lens_type in LensType:
            checkpoint, _ = load_stage4_context_lens_checkpoint(
                project_root=_project_root(),
                stage3_context_artifact=stage3_artifact,
                decision_object_id=obj.decision_object_id,
                lens_type=lens_type,
            )
            if checkpoint is not None:
                checkpoint_outputs[lens_type].append(checkpoint["assessment"])
    return {
        "stage4_context_complete": False,
        "stage4_context_artifact_reused": False,
        "stage4_context_input_fingerprint": identity["input_fingerprint"],
        "stage4_context_artifact_path": str(artifact_path),
        "stage4_context_evaluation_bundle": None,
        "contextual_technical_lens_assessments": checkpoint_outputs[LensType.TECHNICAL_FEASIBILITY],
        "contextual_institutional_lens_assessments": checkpoint_outputs[LensType.INSTITUTIONAL_REGIONAL],
        "contextual_financial_lens_assessments": checkpoint_outputs[LensType.FINANCIAL_ADOPTION],
        "contextual_technical_checkpoint_reuse_count": len(checkpoint_outputs[LensType.TECHNICAL_FEASIBILITY]),
        "contextual_institutional_checkpoint_reuse_count": len(checkpoint_outputs[LensType.INSTITUTIONAL_REGIONAL]),
        "contextual_financial_checkpoint_reuse_count": len(checkpoint_outputs[LensType.FINANCIAL_ADOPTION]),
        "messages": ["**Stage 4 Context**: loaded exact contextual per-lens checkpoints where available."],
    }


async def _run_context_lens(
    state: AgentState,
    *,
    lens_type: LensType,
    state_field: str,
    progress_prefix: str,
) -> dict[str, Any]:
    if state.get("stage4_context_artifact_reused"):
        return {"messages": [f"**{lens_type.value}**: reused frozen contextual assessment(s)."]}
    stage3_artifact = _load_exact_stage3_context_artifact(state)
    objects = [ContextualDecisionObject.model_validate(item) for item in state["contextual_decision_objects"]]
    existing = {str(item["decision_object_id"]): item for item in (state.get(state_field) or [])}
    outputs: list[dict[str, Any]] = []
    for index, obj in enumerate(objects, start=1):
        cached = existing.get(obj.decision_object_id)
        if cached is not None:
            parsed = validate_context_lens_assessment(
                cached,
                decision_object=obj,
                expected_lens_type=lens_type,
            )
            outputs.append(parsed.model_dump(mode="json"))
            continue
        payload = build_context_stage4_evaluation_payload(
            decision_object=obj,
            expected_lens_type=lens_type,
        )
        structured_llm = get_context_llm().with_structured_output(
            ContextLensAssessment,
            method="json_schema",
            progress_label=f"{progress_prefix}-{index}of{len(objects)}-{obj.scenario_id}",
            response_schema=build_constrained_context_lens_response_schema(
                decision_object=obj,
                expected_lens_type=lens_type,
            ),
            semantic_validator=lambda assessment, frozen=obj: validate_context_lens_assessment(
                assessment,
                decision_object=frozen,
                expected_lens_type=lens_type,
            ),
        )
        raw = await structured_llm.ainvoke(
            [
                SystemMessage(content=CONTEXT_SYSTEM_PROMPT_BY_LENS[lens_type.value]),
                HumanMessage(
                    content=CONTEXT_STAGE4_EVALUATION_USER_PROMPT.format(
                        evaluation_payload=json.dumps(
                            payload,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                    )
                ),
            ]
        )
        assessment = raw.model_dump(mode="json")
        checkpoint_path = freeze_stage4_context_lens_checkpoint(
            project_root=_project_root(),
            stage3_context_artifact=stage3_artifact,
            decision_object_id=obj.decision_object_id,
            lens_type=lens_type,
            assessment=assessment,
        )
        print(
            f"Stage 4 v4 {lens_type.value} {obj.scenario_id} checkpoint saved: {checkpoint_path}",
            flush=True,
        )
        outputs.append(assessment)
    return {
        state_field: outputs,
        "messages": [f"**{lens_type.value}**: completed {len(outputs)} contextual scenario assessment(s)."],
    }


_CONTEXT_STATE_FIELD_BY_LENS = {
    LensType.TECHNICAL_FEASIBILITY: "contextual_technical_lens_assessments",
    LensType.INSTITUTIONAL_REGIONAL: "contextual_institutional_lens_assessments",
    LensType.FINANCIAL_ADOPTION: "contextual_financial_lens_assessments",
}

_CONTEXT_PROGRESS_PREFIX_BY_LENS = {
    LensType.TECHNICAL_FEASIBILITY: "Stage4v4A-Technical",
    LensType.INSTITUTIONAL_REGIONAL: "Stage4v4B-InstitutionalRegional",
    LensType.FINANCIAL_ADOPTION: "Stage4v4C-FinancialAdoption",
}


_JURISDICTION_MASK = "JURISDICTION_MASKED_NO_LENS_SPECIFIC_EVIDENCE"


def _equivalence_synthetic_id(
    *,
    decision_object: ContextualDecisionObject,
    lens_type: LensType,
    equivalence_key: tuple[Any, ...],
) -> str:
    digest = hashlib.sha256(
        json.dumps(
            equivalence_key,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:16]
    return f"equivalence__{decision_object.case_id}__{lens_type.value}__{digest}"


def _common_evidence_ids(objects: list[ContextualDecisionObject]) -> set[str]:
    if not objects:
        return set()
    common = set(objects[0].evaluation_evidence_ids)
    for obj in objects[1:]:
        common &= set(obj.evaluation_evidence_ids)
    return common


def _build_jurisdiction_neutral_payload(
    *,
    decision_object: ContextualDecisionObject,
    lens_type: LensType,
    equivalence_members: list[ContextualDecisionObject],
    synthetic_id: str,
) -> tuple[dict[str, Any], set[str]]:
    """Build one region-masked prompt for a jurisdiction-neutral equivalence class."""

    payload = copy.deepcopy(
        build_context_stage4_evaluation_payload(
            decision_object=decision_object,
            expected_lens_type=lens_type,
        )
    )
    common_ids = _common_evidence_ids(equivalence_members)
    if not common_ids:
        raise ValueError("Jurisdiction-neutral Stage 4 equivalence class has no common evidence IDs")

    prompt_obj = payload["decision_object"]
    prompt_obj["decision_object_id"] = synthetic_id
    prompt_obj["deployment_context"]["region"] = _JURISDICTION_MASK
    scenario = prompt_obj["scenario"]
    scenario["scenario_id"] = "JURISDICTION_NEUTRAL_EQUIVALENCE_CLASS"
    scenario["jurisdiction_code"] = "MASKED"
    scenario["region"] = _JURISDICTION_MASK
    scenario["assumptions"] = [
        "The reference enterprise profile is fixed before Stage 4 lens evaluation.",
        "Jurisdiction is intentionally masked because this lens has no jurisdiction-specific evidence in this equivalence class.",
        "No feasibility direction may be inferred from a jurisdiction label.",
    ]

    # A cloned assessment must remain valid for every member. Expose only the
    # evidence IDs shared by the complete equivalence class so a generated
    # claim cannot accidentally cite a scenario-only context record.
    #
    # A shared evaluation universe can still contain jurisdiction-specific
    # background records inherited from the frozen Stage 0 pack (for example
    # an EU primary-law source present in every scenario object). Such a record
    # is common by ID but not jurisdiction-neutral by content. Conservatively
    # exclude it from the neutral prompt when it is not eligible to carry this
    # lens's directional judgment. If a jurisdiction-bearing record *is*
    # directionally eligible, refuse neutral generation instead of silently
    # deleting substantive evidence.
    labels = {
        label
        for member in equivalence_members
        for label in (member.deployment_context.region, member.scenario_id)
        if label
    }
    lens_contract = payload["directional_grounding_contract"][lens_type.value]
    directional_ids = {
        evidence_id
        for scope_contract in lens_contract.values()
        for evidence_id in scope_contract.get("eligible_evidence", {})
    }
    neutral_evidence: list[dict[str, Any]] = []
    for record in payload["evaluation_evidence"]:
        evidence_id = str(record.get("evidence_id"))
        if evidence_id not in common_ids:
            continue
        record_text = json.dumps(record, ensure_ascii=False, sort_keys=True)
        leaked_record_labels = sorted(label for label in labels if label in record_text)
        if leaked_record_labels:
            if evidence_id in directional_ids:
                raise ValueError(
                    "Jurisdiction-neutral Stage 4 equivalence contains jurisdiction-bearing directional evidence: "
                    f"evidence_id={evidence_id}, labels={leaked_record_labels}."
                )
            continue
        neutral_evidence.append(record)

    neutral_ids = {str(record.get("evidence_id")) for record in neutral_evidence}
    if not neutral_ids:
        raise ValueError("Jurisdiction-neutral Stage 4 projection has no label-free common evidence IDs")
    payload["evaluation_evidence"] = neutral_evidence
    payload["evidence_boundary"]["jurisdiction_label_masked"] = True
    payload["evidence_boundary"]["equivalence_class_size"] = len(equivalence_members)

    # Do not silently claim jurisdiction-neutral generation if a region label
    # survives somewhere else in the projected prompt (for example inside a
    # scenario-specific evidence body). Fail before inference instead.
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    leaked_labels: list[str] = []
    for label in sorted(labels):
        if label in serialized and label not in leaked_labels:
            leaked_labels.append(label)
    if leaked_labels:
        raise ValueError(
            "Jurisdiction-neutral Stage 4 projection leaked scenario labels: "
            f"{leaked_labels}. Refusing inference rather than allowing label sensitivity."
        )
    return payload, neutral_ids


def _clone_equivalent_assessment(
    assessment: dict[str, Any],
    *,
    target: ContextualDecisionObject,
    lens_type: LensType,
) -> dict[str, Any]:
    cloned = copy.deepcopy(assessment)
    cloned["decision_object_id"] = target.decision_object_id
    parsed = validate_context_lens_assessment(
        cloned,
        decision_object=target,
        expected_lens_type=lens_type,
    )
    return parsed.model_dump(mode="json")


async def _run_context_lens_for_object(
    state: AgentState,
    *,
    stage3_artifact: dict[str, Any],
    decision_object: ContextualDecisionObject,
    lens_type: LensType,
    cached: dict[str, Any] | None,
    scenario_index: int,
    scenario_count: int,
    equivalence_members: list[ContextualDecisionObject] | None = None,
    equivalence_key: tuple[Any, ...] | None = None,
) -> dict[str, Any]:
    """Run or reuse one lens for exactly one scenario object."""

    if cached is not None:
        parsed = validate_context_lens_assessment(
            cached,
            decision_object=decision_object,
            expected_lens_type=lens_type,
        )
        return parsed.model_dump(mode="json")

    members = equivalence_members or [decision_object]
    jurisdiction_neutral = (
        len(members) > 1
        and equivalence_key is not None
        and equivalence_key[0] == "JURISDICTION_NEUTRAL"
    )
    response_schema = build_constrained_context_lens_response_schema(
        decision_object=decision_object,
        expected_lens_type=lens_type,
    )
    if jurisdiction_neutral:
        synthetic_id = _equivalence_synthetic_id(
            decision_object=decision_object,
            lens_type=lens_type,
            equivalence_key=equivalence_key,
        )
        payload, common_ids = _build_jurisdiction_neutral_payload(
            decision_object=decision_object,
            lens_type=lens_type,
            equivalence_members=members,
            synthetic_id=synthetic_id,
        )
        response_schema["properties"]["decision_object_id"] = {
            "type": "string",
            "const": synthetic_id,
        }
        response_schema["$defs"]["LensClaim"]["properties"]["evidence_ids"]["items"] = {
            "type": "string",
            "enum": sorted(common_ids),
        }
    else:
        payload = build_context_stage4_evaluation_payload(
            decision_object=decision_object,
            expected_lens_type=lens_type,
        )

    def semantic_validator(assessment: ContextLensAssessment) -> LensAssessment:
        value = assessment.model_dump(mode="json")
        if jurisdiction_neutral:
            value["decision_object_id"] = decision_object.decision_object_id
        return validate_context_lens_assessment(
            value,
            decision_object=decision_object,
            expected_lens_type=lens_type,
        )

    structured_llm = get_context_llm().with_structured_output(
        ContextLensAssessment,
        method="json_schema",
        progress_label=(
            f"{_CONTEXT_PROGRESS_PREFIX_BY_LENS[lens_type]}-"
            f"{scenario_index}of{scenario_count}-"
            f"{'JURISDICTION_NEUTRAL' if jurisdiction_neutral else decision_object.scenario_id}"
        ),
        response_schema=response_schema,
        semantic_validator=semantic_validator,
    )
    raw = await structured_llm.ainvoke(
        [
            SystemMessage(content=CONTEXT_SYSTEM_PROMPT_BY_LENS[lens_type.value]),
            HumanMessage(
                content=CONTEXT_STAGE4_EVALUATION_USER_PROMPT.format(
                    evaluation_payload=json.dumps(
                        payload,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
            ),
        ]
    )
    assessment = raw.model_dump(mode="json")
    checkpoint_path = freeze_stage4_context_lens_checkpoint(
        project_root=_project_root(),
        stage3_context_artifact=stage3_artifact,
        decision_object_id=decision_object.decision_object_id,
        lens_type=lens_type,
        assessment=assessment,
    )
    print(
        f"Stage 4 v4 {lens_type.value} {decision_object.scenario_id} checkpoint saved: {checkpoint_path}",
        flush=True,
    )
    return assessment


async def contextual_priority_queue_lenses_node(state: AgentState) -> AgentState:
    """Run contextual lenses from one global KR -> EU -> US priority queue.

    Jobs are enqueued scenario-major and lens-minor, while a fixed two-worker
    pool keeps both inference slots busy. This guarantees that every KR job is
    queued ahead of every EU job without forcing an idle one-slot tail after
    each three-lens scenario.
    """

    if state.get("stage4_context_artifact_reused"):
        return {"messages": ["**Stage 4 Context**: reused frozen priority-queue assessments."]}

    stage3_artifact = _load_exact_stage3_context_artifact(state)
    objects = [ContextualDecisionObject.model_validate(item) for item in state["contextual_decision_objects"]]
    existing_by_lens: dict[LensType, dict[str, dict[str, Any]]] = {}
    for lens_type, state_field in _CONTEXT_STATE_FIELD_BY_LENS.items():
        existing_by_lens[lens_type] = {
            str(item["decision_object_id"]): item
            for item in (state.get(state_field) or [])
        }

    output_maps: dict[LensType, dict[str, dict[str, Any]]] = {
        lens_type: {} for lens_type in LensType
    }
    pending: asyncio.Queue[
        tuple[
            int,
            ContextualDecisionObject,
            LensType,
            list[ContextualDecisionObject],
            tuple[Any, ...],
        ]
    ] = asyncio.Queue()

    groups: dict[
        tuple[LensType, tuple[Any, ...]],
        list[tuple[int, ContextualDecisionObject]],
    ] = {}
    group_order: list[tuple[LensType, tuple[Any, ...]]] = []
    for scenario_index, obj in enumerate(objects, start=1):
        for lens_type in LensType:
            key = context_lens_equivalence_key(obj, lens_type)
            group_id = (lens_type, key)
            if group_id not in groups:
                groups[group_id] = []
                group_order.append(group_id)
            groups[group_id].append((scenario_index, obj))

    for lens_type, key in group_order:
        indexed_members = groups[(lens_type, key)]
        scenario_index, canonical = indexed_members[0]
        members = [obj for _, obj in indexed_members]
        cached = existing_by_lens[lens_type].get(canonical.decision_object_id)
        if cached is not None:
            parsed = validate_context_lens_assessment(
                cached,
                decision_object=canonical,
                expected_lens_type=lens_type,
            ).model_dump(mode="json")
            for member in members:
                output_maps[lens_type][member.decision_object_id] = _clone_equivalent_assessment(
                    parsed,
                    target=member,
                    lens_type=lens_type,
                )
        else:
            pending.put_nowait((scenario_index, canonical, lens_type, members, key))

    print(
        "Stage 4 priority queue: jurisdiction-neutral equivalent scenario/lens inputs are deduplicated before "
        "generation; KR-first canonical jobs are ahead of any jurisdiction-bound EU/US jobs; "
        f"{STAGE4_CLIENT_CONCURRENCY} workers keep the inference slots occupied. "
        f"canonical_jobs={pending.qsize()} from raw_jobs={len(objects) * len(LensType)}.",
        flush=True,
    )

    async def worker(worker_index: int) -> None:
        while True:
            try:
                scenario_index, obj, lens_type, members, key = pending.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                print(
                    f"Stage 4 worker {worker_index}: starting {obj.scenario_id} / {lens_type.value}",
                    flush=True,
                )
                assessment = await _run_context_lens_for_object(
                    state,
                    stage3_artifact=stage3_artifact,
                    decision_object=obj,
                    lens_type=lens_type,
                    cached=None,
                    scenario_index=scenario_index,
                    scenario_count=len(objects),
                    equivalence_members=members,
                    equivalence_key=key,
                )
                for member in members:
                    output_maps[lens_type][member.decision_object_id] = _clone_equivalent_assessment(
                        assessment,
                        target=member,
                        lens_type=lens_type,
                    )
            finally:
                pending.task_done()

    await asyncio.gather(
        *[worker(index + 1) for index in range(STAGE4_CLIENT_CONCURRENCY)]
    )

    outputs: dict[LensType, list[dict[str, Any]]] = {
        lens_type: [
            output_maps[lens_type][obj.decision_object_id]
            for obj in objects
        ]
        for lens_type in LensType
    }

    return {
        "contextual_technical_lens_assessments": outputs[LensType.TECHNICAL_FEASIBILITY],
        "contextual_institutional_lens_assessments": outputs[LensType.INSTITUTIONAL_REGIONAL],
        "contextual_financial_lens_assessments": outputs[LensType.FINANCIAL_ADOPTION],
        "messages": [
            "**Stage 4 Context**: completed KR-priority queued evaluation with isolated sibling lenses."
        ],
    }


async def contextual_technical_feasibility_lens_node(state: AgentState) -> AgentState:
    return await _run_context_lens(
        state,
        lens_type=LensType.TECHNICAL_FEASIBILITY,
        state_field="contextual_technical_lens_assessments",
        progress_prefix="Stage4v4A-Technical",
    )


async def contextual_institutional_regional_lens_node(state: AgentState) -> AgentState:
    return await _run_context_lens(
        state,
        lens_type=LensType.INSTITUTIONAL_REGIONAL,
        state_field="contextual_institutional_lens_assessments",
        progress_prefix="Stage4v4B-InstitutionalRegional",
    )


async def contextual_financial_adoption_lens_node(state: AgentState) -> AgentState:
    return await _run_context_lens(
        state,
        lens_type=LensType.FINANCIAL_ADOPTION,
        state_field="contextual_financial_lens_assessments",
        progress_prefix="Stage4v4C-FinancialAdoption",
    )


async def stage4_context_complete_node(state: AgentState) -> AgentState:
    if state.get("stage4_context_artifact_reused"):
        if not state.get("stage4_context_evaluation_bundle"):
            raise ValueError("Reused contextual Stage 4 artifact is missing its evaluation bundle")
        return {
            "stage4_context_complete": True,
            "messages": [f"**Stage 4 Context**: reused immutable artifact {state.get('stage4_context_artifact_path')}."],
        }

    stage3_artifact = _load_exact_stage3_context_artifact(state)
    objects = [ContextualDecisionObject.model_validate(item) for item in state["contextual_decision_objects"]]
    lists = {
        LensType.TECHNICAL_FEASIBILITY: state.get("contextual_technical_lens_assessments") or [],
        LensType.INSTITUTIONAL_REGIONAL: state.get("contextual_institutional_lens_assessments") or [],
        LensType.FINANCIAL_ADOPTION: state.get("contextual_financial_lens_assessments") or [],
    }
    expected_object_ids = {obj.decision_object_id for obj in objects}
    for lens, assessments in lists.items():
        assessment_ids = [str(item.get("decision_object_id") or "") for item in assessments]
        if len(assessments) != len(objects):
            raise ValueError(f"Contextual Stage 4 requires exactly one {lens.value} assessment per scenario")
        if len(assessment_ids) != len(set(assessment_ids)):
            raise ValueError(f"Contextual Stage 4 rejected duplicate {lens.value} decision_object_id values")
        if set(assessment_ids) != expected_object_ids:
            raise ValueError(f"Contextual Stage 4 {lens.value} assessment object set mismatch")
    maps = {
        lens: {str(item["decision_object_id"]): item for item in assessments}
        for lens, assessments in lists.items()
    }

    evaluations = [
        DecisionLensEvaluation(
            decision_object_id=obj.decision_object_id,
            technical_feasibility=maps[LensType.TECHNICAL_FEASIBILITY][obj.decision_object_id],
            institutional_regional=maps[LensType.INSTITUTIONAL_REGIONAL][obj.decision_object_id],
            financial_adoption=maps[LensType.FINANCIAL_ADOPTION][obj.decision_object_id],
        )
        for obj in objects
    ]
    bundle = validate_context_stage4_bundle(
        Stage4EvaluationBundle(case_id=objects[0].case_id, evaluations=evaluations),
        decision_objects=objects,
    )
    artifact, path = freeze_stage4_context_artifact(
        project_root=_project_root(),
        stage3_context_artifact=stage3_artifact,
        evaluation_bundle=bundle,
    )
    clear_stage4_context_lens_checkpoints(
        project_root=_project_root(),
        stage3_context_artifact=stage3_artifact,
    )
    return {
        "stage4_context_complete": True,
        "stage4_context_artifact_reused": False,
        "stage4_context_input_fingerprint": artifact["input_fingerprint"],
        "stage4_context_artifact_path": str(path),
        "stage4_context_evaluation_bundle": artifact["evaluation_bundle"],
        "messages": [
            f"**Stage 4 Context**: froze {len(evaluations)} scenario-conditioned three-lens evaluation(s) at {path}."
        ],
    }

