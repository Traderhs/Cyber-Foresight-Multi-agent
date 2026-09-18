from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import pvariance
from typing import Any, Iterable

from langchain_core.messages import HumanMessage, SystemMessage

from Stage1.runtime import (
    LlamaCppChatClient,
    assert_llama_server_ready,
    get_runtime_config,
)
from Stage3.context_schema import ContextualDecisionObject
from Stage3.schema import DecisionObject
from Stage4.context_input import build_context_stage4_evaluation_payload
from Stage4.input import build_stage4_evaluation_payload
from Stage4.context_nodes import (
    _build_jurisdiction_neutral_payload,
    _clone_equivalent_assessment,
    _equivalence_synthetic_id,
)
from Stage4.context_prompts import (
    CONTEXT_STAGE4_EVALUATION_USER_PROMPT,
    CONTEXT_SYSTEM_PROMPT_BY_LENS,
)
from Stage4.prompts import STAGE4_EVALUATION_USER_PROMPT, SYSTEM_PROMPT_BY_LENS
from Stage4.schema import (
    LensAssessment,
    build_constrained_lens_response_schema,
    validate_lens_assessment,
)
from Stage4.context_schema import (
    ContextLensAssessment,
    build_constrained_context_lens_response_schema,
    context_lens_equivalence_key,
    validate_context_lens_assessment,
)
from Stage4.schema import DecisionLensEvaluation, LensType
from Stage6.builder import _decision_recommendation
from Stage7.config import (
    SENSITIVITY_CASE_IDS,
    STAGE7_VARIANT_RECORD_VERSION,
)
from Stage7.loaders import CaseBundle, sha256_json, sha256_text
from Stage7.prompts import (
    SINGLE_AGENT_THREE_LENS_SYSTEM_PROMPT,
    SINGLE_AGENT_THREE_LENS_USER_PROMPT,
)
from Stage7.schema import SingleAgentThreeLensResponse, VariantResultRecord


STAGE7_DECISION_VARIANT_RUNNER_VERSION = "stage7-decision-variant-runner-v3"
FROZEN_A1_VARIANT_RUNNER_VERSION = "stage7-decision-variant-runner-v1"
STAGE7_VARIANT_CLIENT_CONCURRENCY = 2
CONTEXT_FREE_VARIANT_ID = "CONTEXT_FREE_STAGE4_V1"

# P1 is the frozen main prompt and is represented by the main artifact rather than
# regenerated. P2/P3 preserve the exact evidence/schema boundary while changing
# instruction order only. The text is part of the variant identity and is frozen.
_PROMPT_PREFIX_BY_VARIANT = {
    "P2_EVIDENCE_FIRST": (
        "Sensitivity wording P2 (evidence-first): before choosing the lens stance, first inventory the supplied "
        "evaluation_evidence and directional_grounding_contract, identify what each exact record can and cannot "
        "support, then answer the same feasibility question under the unchanged schema and rules. Do not add, "
        "remove, reinterpret, or reorder the factual evidence boundary.\n\n"
    ),
    "P3_DECISION_FIRST": (
        "Sensitivity wording P3 (decision-first): first restate internally the exact assigned feasibility question "
        "for this lens, then test each possible direction against the supplied evaluation_evidence and "
        "directional_grounding_contract before returning the same structured answer. Do not add, remove, "
        "reinterpret, or reorder the factual evidence boundary.\n\n"
    ),
}


def _runtime_profile(config, *, runner_version: str) -> dict[str, Any]:
    raw = asdict(config)
    raw.pop("api_base", None)
    raw.pop("timeout_seconds", None)
    raw["expected_context_size"] = 65536
    raw["expected_parallel_slots"] = 2
    raw["server_total_context_size"] = 131072
    raw["client_request_concurrency"] = STAGE7_VARIANT_CLIENT_CONCURRENCY
    raw["variant_runner_version"] = runner_version
    return raw


def _variant_client(
    *,
    runner_version: str = STAGE7_DECISION_VARIANT_RUNNER_VERSION,
) -> tuple[LlamaCppChatClient, dict[str, Any]]:
    config = get_runtime_config()
    assert_llama_server_ready(
        expected_context_size=65536,
        expected_parallel_slots=2,
        profile_name="Stage 7 final variants 128K-total / 2-slot",
    )
    return (
        LlamaCppChatClient(config=config, request_concurrency=STAGE7_VARIANT_CLIENT_CONCURRENCY),
        _runtime_profile(config, runner_version=runner_version),
    )


def _prompt_texts(*, lens_type: LensType, prompt_variant: str, payload: dict[str, Any]) -> tuple[str, str]:
    system_prompt = CONTEXT_SYSTEM_PROMPT_BY_LENS[lens_type.value]
    user_prompt = CONTEXT_STAGE4_EVALUATION_USER_PROMPT.format(
        evaluation_payload=json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )
    if prompt_variant == "P1_CANONICAL":
        return system_prompt, user_prompt
    prefix = _PROMPT_PREFIX_BY_VARIANT.get(prompt_variant)
    if prefix is None:
        raise ValueError(f"Unknown prompt sensitivity variant: {prompt_variant}")
    return system_prompt, prefix + user_prompt


def _checkpoint_path(root: Path, *, axis: str, variant_id: str, fingerprint: str) -> Path:
    return (
        root
        / "Multi-Agent"
        / "Results"
        / "Stage7"
        / "variant_checkpoints"
        / axis
        / variant_id
        / f"{fingerprint}.json"
    )


def _load_checkpoint(path: Path, identity: dict[str, Any]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    stored = str(value.get("checkpoint_sha256") or "")
    unhashed = {k: v for k, v in value.items() if k != "checkpoint_sha256"}
    if not stored or stored != sha256_json(unhashed):
        raise ValueError(f"Stage 7 variant checkpoint hash mismatch: {path}")
    if sha256_json(value.get("identity")) != sha256_json(identity):
        raise ValueError(f"Stage 7 variant checkpoint identity mismatch: {path}")
    return value


def _freeze_checkpoint(
    path: Path,
    *,
    identity: dict[str, Any],
    assessment: dict[str, Any],
) -> None:
    payload = {
        "identity": identity,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "assessment": assessment,
        "assessment_sha256": sha256_json(assessment),
    }
    payload["checkpoint_sha256"] = sha256_json(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        existing = _load_checkpoint(path, identity)
        if existing is None or existing.get("assessment_sha256") != payload["assessment_sha256"]:
            raise ValueError(f"Conflicting Stage 7 variant checkpoint: {path}")
        return
    path.write_text(serialized, encoding="utf-8", newline="\n")


def _variant_output_path(root: Path, *, axis: str, variant_id: str, repeat_index: int | None) -> Path:
    suffix = f"__repeat-{repeat_index}" if repeat_index is not None else ""
    return root / "Multi-Agent" / "Results" / "Stage7" / "variants" / axis / f"{variant_id}{suffix}.json"


def _freeze_variant_records(
    root: Path,
    *,
    axis: str,
    variant_id: str,
    repeat_index: int | None,
    records: list[VariantResultRecord],
    provenance: dict[str, Any],
    runner_version: str = STAGE7_DECISION_VARIANT_RUNNER_VERSION,
) -> Path:
    path = _variant_output_path(root, axis=axis, variant_id=variant_id, repeat_index=repeat_index)
    payload = {
        "schema_version": STAGE7_VARIANT_RECORD_VERSION,
        "runner_version": runner_version,
        "axis": axis,
        "variant_id": variant_id,
        "repeat_index": repeat_index,
        "provenance": provenance,
        "records": [record.model_dump(mode="json") for record in records],
    }
    payload["payload_sha256"] = sha256_json(payload)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        old = {k: v for k, v in existing.items() if k != "created_at"}
        if old != payload:
            raise ValueError(f"Frozen Stage 7 variant result changed in place: {path}")
        return path
    path.write_text(serialized, encoding="utf-8", newline="\n")
    return path


def _evidence_ids(assessment: dict[str, Any]) -> list[str]:
    return list(
        dict.fromkeys(
            evidence_id
            for claim in assessment.get("claims") or []
            for evidence_id in claim.get("evidence_ids") or []
        )
    )


def _directions(assessment: dict[str, Any]) -> list[str]:
    return [str(claim.get("status") or "") for claim in assessment.get("claims") or []]


def _build_single_agent_response_schema(
    obj: ContextualDecisionObject,
    *,
    decision_object_id: str | None = None,
    allowed_evidence_ids: set[str] | None = None,
) -> dict[str, Any]:
    schema = deepcopy(SingleAgentThreeLensResponse.model_json_schema())
    schema["properties"]["decision_object_id"] = {
        "type": "string",
        "const": decision_object_id or obj.decision_object_id,
    }
    lens_def = schema.get("$defs", {}).get("LensAssessment")
    claim_def = schema.get("$defs", {}).get("LensClaim")
    if not isinstance(lens_def, dict) or not isinstance(claim_def, dict):
        raise ValueError("Single-agent response schema is missing Stage4 lens definitions")
    lens_def["required"] = sorted(lens_def.get("properties", {}))
    lens_def["properties"]["claims"]["minItems"] = 1
    claim_def["properties"]["evidence_ids"]["items"] = {
        "type": "string",
        "enum": sorted(allowed_evidence_ids or set(obj.evaluation_evidence_ids)),
    }
    return schema


def _validate_single_agent_response(
    response: SingleAgentThreeLensResponse | dict[str, Any],
    *,
    obj: ContextualDecisionObject,
) -> SingleAgentThreeLensResponse:
    parsed = (
        response
        if isinstance(response, SingleAgentThreeLensResponse)
        else SingleAgentThreeLensResponse.model_validate(response)
    )
    if parsed.decision_object_id != obj.decision_object_id:
        raise ValueError("Single-agent decision_object_id mismatch")
    normalized = parsed.model_dump(mode="json")
    for field_name, lens_type in (
        ("technical_feasibility", LensType.TECHNICAL_FEASIBILITY),
        ("institutional_regional", LensType.INSTITUTIONAL_REGIONAL),
        ("financial_adoption", LensType.FINANCIAL_ADOPTION),
    ):
        normalized[field_name] = validate_context_lens_assessment(
            normalized[field_name],
            decision_object=obj,
            expected_lens_type=lens_type,
        ).model_dump(mode="json")
    return SingleAgentThreeLensResponse.model_validate(normalized)


def _single_agent_equivalence_key(obj: ContextualDecisionObject) -> tuple[Any, ...]:
    return tuple(
        context_lens_equivalence_key(obj, lens_type)
        for lens_type in (
            LensType.TECHNICAL_FEASIBILITY,
            LensType.INSTITUTIONAL_REGIONAL,
            LensType.FINANCIAL_ADOPTION,
        )
    )


def _build_single_agent_neutral_payload(
    *,
    canonical: ContextualDecisionObject,
    members: list[ContextualDecisionObject],
    synthetic_id: str,
) -> tuple[dict[str, Any], set[str]]:
    """Merge the three main lens-neutral projections into one joint-agent prompt."""

    payloads: list[dict[str, Any]] = []
    evidence_by_id: dict[str, dict[str, Any]] = {}
    contracts: dict[str, Any] = {}
    allowed_ids: set[str] = set()
    for lens_type in (
        LensType.TECHNICAL_FEASIBILITY,
        LensType.INSTITUTIONAL_REGIONAL,
        LensType.FINANCIAL_ADOPTION,
    ):
        payload, lens_ids = _build_jurisdiction_neutral_payload(
            decision_object=canonical,
            lens_type=lens_type,
            equivalence_members=members,
            synthetic_id=synthetic_id,
        )
        payloads.append(payload)
        for record in payload["evaluation_evidence"]:
            evidence_by_id[str(record["evidence_id"])] = record
        contracts.update(payload["directional_grounding_contract"])
        allowed_ids.update(lens_ids)
    merged = deepcopy(payloads[0])
    merged["evaluation_evidence"] = [
        evidence_by_id[evidence_id] for evidence_id in sorted(evidence_by_id)
    ]
    merged["directional_grounding_contract"] = contracts
    merged["evidence_boundary"]["joint_three_lens_single_agent"] = True
    merged["evidence_boundary"]["sibling_lens_outputs_visible"] = True
    return merged, allowed_ids


def _clone_single_agent_response(
    response: SingleAgentThreeLensResponse,
    *,
    target: ContextualDecisionObject,
) -> SingleAgentThreeLensResponse:
    value = response.model_dump(mode="json")
    value["decision_object_id"] = target.decision_object_id
    for field_name, lens_type in (
        ("technical_feasibility", LensType.TECHNICAL_FEASIBILITY),
        ("institutional_regional", LensType.INSTITUTIONAL_REGIONAL),
        ("financial_adoption", LensType.FINANCIAL_ADOPTION),
    ):
        value[field_name] = _clone_equivalent_assessment(
            value[field_name],
            target=target,
            lens_type=lens_type,
        )
    return _validate_single_agent_response(value, obj=target)


async def run_single_agent_baseline(
    root: Path,
    bundles: list[CaseBundle],
) -> Path:
    """Run one joint three-lens agent while preserving main jurisdiction equivalence."""

    client, runtime = _variant_client()
    pending: asyncio.Queue[
        tuple[CaseBundle, tuple[Any, ...], list[ContextualDecisionObject]]
    ] = asyncio.Queue()
    identities: dict[
        tuple[str, str],
        tuple[dict[str, Any], Path, str, str, dict[str, Any], bool],
    ] = {}
    outputs: dict[tuple[str, str], SingleAgentThreeLensResponse] = {}
    canonical_job_count = 0

    for bundle in bundles:
        objects = [ContextualDecisionObject.model_validate(raw) for raw in bundle.contextual_objects]
        groups: dict[tuple[Any, ...], list[ContextualDecisionObject]] = {}
        for obj in objects:
            groups.setdefault(_single_agent_equivalence_key(obj), []).append(obj)

        for group_key, members in groups.items():
            canonical = members[0]
            jurisdiction_neutral = (
                len(members) > 1
                and all(
                    isinstance(lens_key, tuple)
                    and lens_key
                    and lens_key[0] == "JURISDICTION_NEUTRAL"
                    for lens_key in group_key
                )
            )
            group_hash = sha256_json(group_key)[:16]
            if jurisdiction_neutral:
                synthetic_id = f"single_agent__{bundle.case_id}__{group_hash}"
                payload, allowed_ids = _build_single_agent_neutral_payload(
                    canonical=canonical,
                    members=members,
                    synthetic_id=synthetic_id,
                )
                response_schema = _build_single_agent_response_schema(
                    canonical,
                    decision_object_id=synthetic_id,
                    allowed_evidence_ids=allowed_ids,
                )
            else:
                payload = build_context_stage4_evaluation_payload(
                    decision_object=canonical,
                    expected_lens_type=None,
                )
                response_schema = _build_single_agent_response_schema(canonical)
            system_prompt = SINGLE_AGENT_THREE_LENS_SYSTEM_PROMPT
            user_prompt = SINGLE_AGENT_THREE_LENS_USER_PROMPT.format(
                evaluation_payload=json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            identity = {
                "runner_version": STAGE7_DECISION_VARIANT_RUNNER_VERSION,
                "axis": "B_STAGE4_ARCHITECTURE_BASELINE",
                "variant_id": "SINGLE_AGENT_BASELINE",
                "case_id": bundle.case_id,
                "equivalence_key": group_key,
                "member_decision_object_ids": [member.decision_object_id for member in members],
                "jurisdiction_neutral": jurisdiction_neutral,
                "payload_sha256": sha256_json(payload),
                "system_prompt_sha256": sha256_text(system_prompt),
                "user_prompt_sha256": sha256_text(user_prompt),
                "response_schema_sha256": sha256_json(response_schema),
                "runtime": runtime,
                "source_stage3_context_artifact_sha256": bundle.stage3_context["artifact_sha256"],
            }
            identity["input_fingerprint"] = sha256_json(identity)
            path = _checkpoint_path(
                root,
                axis="B_STAGE4_ARCHITECTURE_BASELINE",
                variant_id="SINGLE_AGENT_BASELINE",
                fingerprint=identity["input_fingerprint"],
            )
            identity_key = (bundle.case_id, group_hash)
            identities[identity_key] = (
                identity,
                path,
                system_prompt,
                user_prompt,
                response_schema,
                jurisdiction_neutral,
            )
            checkpoint = _load_checkpoint(path, identity)
            if checkpoint is not None:
                canonical_response = _validate_single_agent_response(
                    checkpoint["assessment"], obj=canonical
                )
                for member in members:
                    outputs[(bundle.case_id, member.scenario_id)] = _clone_single_agent_response(
                        canonical_response,
                        target=member,
                    )
            else:
                pending.put_nowait((bundle, group_key, members))
            canonical_job_count += 1

    print(
        f"Stage 7 B/SINGLE_AGENT_BASELINE: {pending.qsize()} missing canonical jobs "
        f"from {canonical_job_count}; {STAGE7_VARIANT_CLIENT_CONCURRENCY} workers / 2 llama.cpp slots.",
        flush=True,
    )

    async def worker(worker_index: int) -> None:
        while True:
            try:
                bundle, group_key, members = pending.get_nowait()
            except asyncio.QueueEmpty:
                return
            canonical = members[0]
            group_hash = sha256_json(group_key)[:16]
            identity, path, system_prompt, user_prompt, response_schema, jurisdiction_neutral = identities[
                (bundle.case_id, group_hash)
            ]

            def semantic_validator(response: SingleAgentThreeLensResponse):
                value = response.model_dump(mode="json")
                if jurisdiction_neutral:
                    value["decision_object_id"] = canonical.decision_object_id
                    for field_name in (
                        "technical_feasibility",
                        "institutional_regional",
                        "financial_adoption",
                    ):
                        value[field_name]["decision_object_id"] = canonical.decision_object_id
                return _validate_single_agent_response(value, obj=canonical)

            runner = client.with_structured_output(
                SingleAgentThreeLensResponse,
                method="json_schema",
                progress_label=(
                    f"Stage7-B-Single-W{worker_index}-{bundle.case_id}-{group_hash}"
                ),
                response_schema=response_schema,
                semantic_validator=semantic_validator,
            )
            try:
                response = await runner.ainvoke(
                    [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
                )
                response_dict = response.model_dump(mode="json")
                _freeze_checkpoint(path, identity=identity, assessment=response_dict)
                for member in members:
                    outputs[(bundle.case_id, member.scenario_id)] = _clone_single_agent_response(
                        response,
                        target=member,
                    )
            finally:
                pending.task_done()

    await asyncio.gather(
        *[worker(index + 1) for index in range(STAGE7_VARIANT_CLIENT_CONCURRENCY)]
    )

    records: list[VariantResultRecord] = []
    for bundle in bundles:
        for raw in bundle.contextual_objects:
            obj = ContextualDecisionObject.model_validate(raw)
            response = outputs[(bundle.case_id, obj.scenario_id)]
            by_lens = {
                LensType.TECHNICAL_FEASIBILITY: response.technical_feasibility.model_dump(mode="json"),
                LensType.INSTITUTIONAL_REGIONAL: response.institutional_regional.model_dump(mode="json"),
                LensType.FINANCIAL_ADOPTION: response.financial_adoption.model_dump(mode="json"),
            }
            evaluation = DecisionLensEvaluation(
                decision_object_id=obj.decision_object_id,
                technical_feasibility=by_lens[LensType.TECHNICAL_FEASIBILITY],
                institutional_regional=by_lens[LensType.INSTITUTIONAL_REGIONAL],
                financial_adoption=by_lens[LensType.FINANCIAL_ADOPTION],
            )
            recommendation, _ = _decision_recommendation(evaluation=evaluation)
            stances = {
                lens.value: int(assessment["stance"])
                for lens, assessment in by_lens.items()
            }
            records.append(
                VariantResultRecord(
                    axis="B_STAGE4_ARCHITECTURE_BASELINE",
                    variant_id="SINGLE_AGENT_BASELINE",
                    case_id=bundle.case_id,
                    scenario_id=obj.scenario_id,
                    decision_object_id=obj.decision_object_id,
                    stance_by_lens=stances,
                    d_lens=round(float(pvariance(list(stances.values()))), 6),
                    evidence_ids_by_lens={
                        lens.value: _evidence_ids(assessment)
                        for lens, assessment in by_lens.items()
                    },
                    claim_directions_by_lens={
                        lens.value: _directions(assessment)
                        for lens, assessment in by_lens.items()
                    },
                    recommendation=recommendation.value,
                    prompt_hashes={
                        "system_prompt_sha256": sha256_text(SINGLE_AGENT_THREE_LENS_SYSTEM_PROMPT),
                    },
                    runtime=runtime,
                    provenance={
                        "runner_version": STAGE7_DECISION_VARIANT_RUNNER_VERSION,
                        "single_agent_joint_three_lens": True,
                        "sibling_lens_information_isolation": False,
                        "jurisdiction_equivalence_dedup_preserved": True,
                        "source_stage3_context_artifact_sha256": bundle.stage3_context["artifact_sha256"],
                        "source_stage6_artifact_sha256": bundle.stage6["artifact_sha256"],
                    },
                )
            )
    return _freeze_variant_records(
        root,
        axis="B_STAGE4_ARCHITECTURE_BASELINE",
        variant_id="SINGLE_AGENT_BASELINE",
        repeat_index=None,
        records=records,
        provenance={
            "case_ids": [bundle.case_id for bundle in bundles],
            "scenario_count": len(records),
            "single_agent_joint_three_lens": True,
            "jurisdiction_equivalence_dedup_preserved": True,
            "canonical_job_count": canonical_job_count,
        },
    )


def _records_from_assessments(
    *,
    axis: str,
    variant_id: str,
    repeat_index: int | None,
    bundle: CaseBundle,
    objects: list[ContextualDecisionObject],
    assessment_by_object_lens: dict[tuple[str, LensType], dict[str, Any]],
    prompt_hashes: dict[str, str],
    runtime: dict[str, Any],
    provenance: dict[str, Any],
) -> list[VariantResultRecord]:
    records: list[VariantResultRecord] = []
    for obj in objects:
        by_lens = {
            lens_type: assessment_by_object_lens[(obj.decision_object_id, lens_type)]
            for lens_type in LensType
        }
        evaluation = DecisionLensEvaluation(
            decision_object_id=obj.decision_object_id,
            technical_feasibility=by_lens[LensType.TECHNICAL_FEASIBILITY],
            institutional_regional=by_lens[LensType.INSTITUTIONAL_REGIONAL],
            financial_adoption=by_lens[LensType.FINANCIAL_ADOPTION],
        )
        recommendation, _ = _decision_recommendation(evaluation=evaluation)
        stances = {
            lens_type.value: int(assessment["stance"])
            for lens_type, assessment in by_lens.items()
        }
        d_lens = round(float(pvariance(list(stances.values()))), 6)
        records.append(
            VariantResultRecord(
                axis=axis,
                variant_id=variant_id,
                repeat_index=repeat_index,
                case_id=bundle.case_id,
                scenario_id=obj.scenario_id,
                decision_object_id=obj.decision_object_id,
                stance_by_lens=stances,
                d_lens=d_lens,
                evidence_ids_by_lens={
                    lens_type.value: _evidence_ids(assessment)
                    for lens_type, assessment in by_lens.items()
                },
                claim_directions_by_lens={
                    lens_type.value: _directions(assessment)
                    for lens_type, assessment in by_lens.items()
                },
                recommendation=recommendation.value,
                prompt_hashes=prompt_hashes,
                runtime=runtime,
                provenance={
                    **provenance,
                    "source_stage3_context_artifact_sha256": bundle.stage3_context["artifact_sha256"],
                    "source_stage6_artifact_sha256": bundle.stage6["artifact_sha256"],
                },
            )
        )
    return records


async def _run_one_case_variant(
    root: Path,
    *,
    bundle: CaseBundle,
    axis: str,
    variant_id: str,
    repeat_index: int | None,
    prompt_variant: str,
) -> tuple[list[VariantResultRecord], dict[str, Any]]:
    client, runtime = _variant_client(runner_version=FROZEN_A1_VARIANT_RUNNER_VERSION)
    objects = [ContextualDecisionObject.model_validate(item) for item in bundle.contextual_objects]

    # Reproduce the active Stage 4 equivalence-class behavior exactly: same-lens
    # KR/EU/US objects with no lens-specific jurisdiction evidence are generated
    # once from a jurisdiction-masked prompt and then cloned/validated per scenario.
    groups: dict[tuple[LensType, tuple[Any, ...]], list[ContextualDecisionObject]] = {}
    order: list[tuple[LensType, tuple[Any, ...]]] = []
    for obj in objects:
        for lens_type in LensType:
            key = context_lens_equivalence_key(obj, lens_type)
            gid = (lens_type, key)
            if gid not in groups:
                groups[gid] = []
                order.append(gid)
            groups[gid].append(obj)

    pending: asyncio.Queue[tuple[LensType, tuple[Any, ...], list[ContextualDecisionObject]]] = asyncio.Queue()
    output_map: dict[tuple[str, LensType], dict[str, Any]] = {}
    identities: dict[tuple[LensType, tuple[Any, ...]], tuple[dict[str, Any], Path]] = {}
    prompt_hash_accumulator: dict[str, str] = {}

    for lens_type, key in order:
        members = groups[(lens_type, key)]
        canonical = members[0]
        jurisdiction_neutral = len(members) > 1 and key[0] == "JURISDICTION_NEUTRAL"
        response_schema = build_constrained_context_lens_response_schema(
            decision_object=canonical,
            expected_lens_type=lens_type,
        )
        if jurisdiction_neutral:
            synthetic_id = _equivalence_synthetic_id(
                decision_object=canonical,
                lens_type=lens_type,
                equivalence_key=key,
            )
            payload, common_ids = _build_jurisdiction_neutral_payload(
                decision_object=canonical,
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
            synthetic_id = None
            payload = build_context_stage4_evaluation_payload(
                decision_object=canonical,
                expected_lens_type=lens_type,
            )
        system_prompt, user_prompt = _prompt_texts(
            lens_type=lens_type,
            prompt_variant=prompt_variant,
            payload=payload,
        )
        prompt_hash_accumulator[f"{lens_type.value}:{sha256_json(list(key))[:12]}"] = sha256_json(
            {"system": system_prompt, "user": user_prompt}
        )
        identity = {
            "runner_version": FROZEN_A1_VARIANT_RUNNER_VERSION,
            "axis": axis,
            "variant_id": variant_id,
            "repeat_index": repeat_index,
            "case_id": bundle.case_id,
            "lens_type": lens_type.value,
            "equivalence_key": list(key),
            "member_decision_object_ids": [member.decision_object_id for member in members],
            "prompt_variant": prompt_variant,
            # Preserve the frozen v1 checkpoint identity used by the completed
            # A prompt-robustness run. Final Stage 7 uses only the main runtime
            # and no payload transform, but these explicit fields remain part
            # of the immutable identity for exact checkpoint reuse.
            "runtime_variant": "MAIN",
            "payload_transform_id": None,
            "payload_sha256": sha256_json(payload),
            "system_prompt_sha256": sha256_text(system_prompt),
            "user_prompt_sha256": sha256_text(user_prompt),
            "response_schema_sha256": sha256_json(response_schema),
            "runtime": runtime,
            "source_stage3_context_artifact_sha256": bundle.stage3_context["artifact_sha256"],
        }
        identity["input_fingerprint"] = sha256_json(identity)
        path = _checkpoint_path(
            root,
            axis=axis,
            variant_id=variant_id,
            fingerprint=identity["input_fingerprint"],
        )
        identities[(lens_type, key)] = (identity, path)
        checkpoint = _load_checkpoint(path, identity)
        if checkpoint is not None:
            canonical_assessment = dict(checkpoint["assessment"])
            for member in members:
                output_map[(member.decision_object_id, lens_type)] = _clone_equivalent_assessment(
                    canonical_assessment,
                    target=member,
                    lens_type=lens_type,
                )
        else:
            pending.put_nowait((lens_type, key, members))

    print(
        f"Stage 7 {axis}/{variant_id}/{bundle.case_id}: {pending.qsize()} missing canonical jobs; "
        f"{STAGE7_VARIANT_CLIENT_CONCURRENCY} workers / 2 llama.cpp slots.",
        flush=True,
    )

    async def worker(worker_index: int) -> None:
        while True:
            try:
                lens_type, key, members = pending.get_nowait()
            except asyncio.QueueEmpty:
                return
            canonical = members[0]
            identity, path = identities[(lens_type, key)]
            jurisdiction_neutral = len(members) > 1 and key[0] == "JURISDICTION_NEUTRAL"
            response_schema = build_constrained_context_lens_response_schema(
                decision_object=canonical,
                expected_lens_type=lens_type,
            )
            if jurisdiction_neutral:
                synthetic_id = _equivalence_synthetic_id(
                    decision_object=canonical,
                    lens_type=lens_type,
                    equivalence_key=key,
                )
                payload, common_ids = _build_jurisdiction_neutral_payload(
                    decision_object=canonical,
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
                synthetic_id = None
                payload = build_context_stage4_evaluation_payload(
                    decision_object=canonical,
                    expected_lens_type=lens_type,
                )
            system_prompt, user_prompt = _prompt_texts(
                lens_type=lens_type,
                prompt_variant=prompt_variant,
                payload=payload,
            )

            def semantic_validator(assessment: ContextLensAssessment):
                value = assessment.model_dump(mode="json")
                if jurisdiction_neutral:
                    value["decision_object_id"] = canonical.decision_object_id
                return validate_context_lens_assessment(
                    value,
                    decision_object=canonical,
                    expected_lens_type=lens_type,
                )

            runner = client.with_structured_output(
                ContextLensAssessment,
                method="json_schema",
                progress_label=f"Stage7-{axis}-W{worker_index}-{variant_id}-{bundle.case_id}-{lens_type.value}",
                response_schema=response_schema,
                semantic_validator=semantic_validator,
            )
            try:
                raw = await runner.ainvoke(
                    [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
                )
                assessment = raw.model_dump(mode="json")
                _freeze_checkpoint(path, identity=identity, assessment=assessment)
                for member in members:
                    output_map[(member.decision_object_id, lens_type)] = _clone_equivalent_assessment(
                        assessment,
                        target=member,
                        lens_type=lens_type,
                    )
            finally:
                pending.task_done()

    await asyncio.gather(
        *[worker(index + 1) for index in range(STAGE7_VARIANT_CLIENT_CONCURRENCY)]
    )

    expected = len(objects) * len(LensType)
    if len(output_map) != expected:
        raise ValueError(
            f"Incomplete Stage 7 variant outputs for {bundle.case_id}: {len(output_map)} != {expected}"
        )

    prompt_hashes = {
        "prompt_variant": prompt_variant,
        "canonical_job_prompt_set_sha256": sha256_json(prompt_hash_accumulator),
    }
    provenance = {
        "runner_version": FROZEN_A1_VARIANT_RUNNER_VERSION,
        "runtime_variant": "MAIN",
        "prompt_variant": prompt_variant,
        "canonical_job_count": len(order),
        "raw_scenario_lens_count": expected,
        "jurisdiction_equivalence_dedup": True,
    }
    return (
        _records_from_assessments(
            axis=axis,
            variant_id=variant_id,
            repeat_index=repeat_index,
            bundle=bundle,
            objects=objects,
            assessment_by_object_lens=output_map,
            prompt_hashes=prompt_hashes,
            runtime=runtime,
            provenance=provenance,
        ),
        provenance,
    )


def _selected_bundles(bundles: Iterable[CaseBundle]) -> list[CaseBundle]:
    by_id = {bundle.case_id: bundle for bundle in bundles}
    missing = [case_id for case_id in SENSITIVITY_CASE_IDS if case_id not in by_id]
    if missing:
        raise ValueError(f"Stage 7 sensitivity subset missing frozen main case(s): {missing}")
    return [by_id[case_id] for case_id in SENSITIVITY_CASE_IDS]


async def run_prompt_paraphrase_variants(root: Path, bundles: list[CaseBundle]) -> list[Path]:
    paths: list[Path] = []
    selected = _selected_bundles(bundles)
    for variant_id in ("P2_EVIDENCE_FIRST", "P3_DECISION_FIRST"):
        records: list[VariantResultRecord] = []
        provenance_by_case: dict[str, Any] = {}
        for bundle in selected:
            case_records, provenance = await _run_one_case_variant(
                root,
                bundle=bundle,
                axis="A1_PROMPT_PARAPHRASE",
                variant_id=variant_id,
                repeat_index=None,
                prompt_variant=variant_id,
            )
            records.extend(case_records)
            provenance_by_case[bundle.case_id] = provenance
        paths.append(
            _freeze_variant_records(
                root,
                axis="A1_PROMPT_PARAPHRASE",
                variant_id=variant_id,
                repeat_index=None,
                records=records,
                provenance={"case_ids": list(SENSITIVITY_CASE_IDS), "by_case": provenance_by_case},
                runner_version=FROZEN_A1_VARIANT_RUNNER_VERSION,
            )
        )
    return paths


def _load_base_stage3_decision_object(root: Path, bundle: CaseBundle) -> tuple[DecisionObject, dict[str, Any]]:
    fingerprint = str(bundle.stage3_context.get("base_stage3_input_fingerprint") or "")
    expected_artifact_sha = str(bundle.stage3_context.get("base_stage3_artifact_sha256") or "")
    expected_output_sha = str(bundle.stage3_context.get("base_stage3_output_sha256") or "")
    if not fingerprint or not expected_artifact_sha or not expected_output_sha:
        raise ValueError(f"Contextual Stage 3 artifact is missing base Stage 3 lineage: {bundle.case_id}")
    path = root / "Multi-Agent" / "Results" / "Stage3" / "artifacts" / bundle.case_id / f"{fingerprint}.json"
    if not path.exists():
        raise FileNotFoundError(f"Exact base Stage 3 artifact missing for C2: {path}")
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if str(artifact.get("input_fingerprint")) != fingerprint:
        raise ValueError(f"Base Stage 3 fingerprint mismatch for C2: {bundle.case_id}")
    if str(artifact.get("artifact_sha256")) != expected_artifact_sha:
        raise ValueError(f"Base Stage 3 artifact hash mismatch for C2: {bundle.case_id}")
    if str(artifact.get("stage3_output_sha256")) != expected_output_sha:
        raise ValueError(f"Base Stage 3 output hash mismatch for C2: {bundle.case_id}")
    objects = list((artifact.get("decision_bundle") or {}).get("decision_objects") or [])
    if len(objects) != 1:
        raise ValueError(f"C2 expects exactly one base DecisionObject per frozen main case: {bundle.case_id}")
    obj = DecisionObject.model_validate(objects[0])
    contextual_forecast_ids = set(bundle.contextual_objects[0].get("forecast_evidence_ids") or [])
    if set(obj.evaluation_evidence_ids) != contextual_forecast_ids:
        raise ValueError(
            f"C2 base Stage 3 evidence does not equal contextual forecast evidence for {bundle.case_id}"
        )
    return obj, artifact


def _context_free_evidence_pack(bundle: CaseBundle, obj: DecisionObject) -> dict[str, Any]:
    all_records = bundle.evidence_records_by_id()
    missing = sorted(set(obj.evaluation_evidence_ids) - set(all_records))
    if missing:
        raise ValueError(f"C2 missing exact forecast evidence records for {bundle.case_id}: {missing}")
    records = [all_records[evidence_id] for evidence_id in sorted(obj.evaluation_evidence_ids)]
    return {
        "case_id": bundle.case_id,
        "source_snapshot_id": str(bundle.stage1.get("source_snapshot_id") or ""),
        "evidence": records,
    }


async def _run_context_free_case(
    root: Path,
    *,
    bundle: CaseBundle,
) -> tuple[list[VariantResultRecord], dict[str, Any]]:
    client, runtime = _variant_client()
    obj, stage3_artifact = _load_base_stage3_decision_object(root, bundle)
    evidence_pack = _context_free_evidence_pack(bundle, obj)
    pending: asyncio.Queue[LensType] = asyncio.Queue()
    outputs: dict[LensType, dict[str, Any]] = {}
    identities: dict[LensType, tuple[dict[str, Any], Path]] = {}
    prompt_hashes: dict[str, str] = {}

    for lens_type in LensType:
        payload = build_stage4_evaluation_payload(
            decision_object=obj,
            evidence_pack=evidence_pack,
        )
        system_prompt = SYSTEM_PROMPT_BY_LENS[lens_type.value]
        user_prompt = STAGE4_EVALUATION_USER_PROMPT.format(
            evaluation_payload=json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        )
        response_schema = build_constrained_lens_response_schema(
            decision_object=obj,
            evidence_pack=evidence_pack,
            expected_lens_type=lens_type,
        )
        identity = {
            "runner_version": STAGE7_DECISION_VARIANT_RUNNER_VERSION,
            "axis": "C2_CONTEXTUALIZED_PATH_COMPARISON",
            "variant_id": CONTEXT_FREE_VARIANT_ID,
            "case_id": bundle.case_id,
            "lens_type": lens_type.value,
            "base_stage3_input_fingerprint": stage3_artifact["input_fingerprint"],
            "base_stage3_artifact_sha256": stage3_artifact["artifact_sha256"],
            "base_stage3_output_sha256": stage3_artifact["stage3_output_sha256"],
            "evidence_ids": sorted(obj.evaluation_evidence_ids),
            "system_prompt_sha256": sha256_text(system_prompt),
            "user_prompt_sha256": sha256_text(user_prompt),
            "response_schema_sha256": sha256_json(response_schema),
            "runtime": runtime,
        }
        identity["input_fingerprint"] = sha256_json(identity)
        path = _checkpoint_path(
            root,
            axis="C2_CONTEXTUALIZED_PATH_COMPARISON",
            variant_id=CONTEXT_FREE_VARIANT_ID,
            fingerprint=identity["input_fingerprint"],
        )
        identities[lens_type] = (identity, path)
        prompt_hashes[lens_type.value] = sha256_json({"system": system_prompt, "user": user_prompt})
        checkpoint = _load_checkpoint(path, identity)
        if checkpoint is not None:
            outputs[lens_type] = dict(checkpoint["assessment"])
        else:
            pending.put_nowait(lens_type)

    print(
        f"Stage 7 C2/{CONTEXT_FREE_VARIANT_ID}/{bundle.case_id}: {pending.qsize()} missing lens jobs; "
        "2 workers / 2 llama.cpp slots.",
        flush=True,
    )

    async def worker(worker_index: int) -> None:
        while True:
            try:
                lens_type = pending.get_nowait()
            except asyncio.QueueEmpty:
                return
            identity, path = identities[lens_type]
            payload = build_stage4_evaluation_payload(decision_object=obj, evidence_pack=evidence_pack)
            system_prompt = SYSTEM_PROMPT_BY_LENS[lens_type.value]
            user_prompt = STAGE4_EVALUATION_USER_PROMPT.format(
                evaluation_payload=json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            )
            response_schema = build_constrained_lens_response_schema(
                decision_object=obj,
                evidence_pack=evidence_pack,
                expected_lens_type=lens_type,
            )

            def semantic_validator(assessment: LensAssessment):
                return validate_lens_assessment(
                    assessment,
                    decision_object=obj,
                    evidence_pack=evidence_pack,
                    expected_lens_type=lens_type,
                )

            runner = client.with_structured_output(
                LensAssessment,
                method="json_schema",
                progress_label=f"Stage7-C2-W{worker_index}-{bundle.case_id}-{lens_type.value}",
                response_schema=response_schema,
                semantic_validator=semantic_validator,
            )
            try:
                result = await runner.ainvoke(
                    [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
                )
                assessment = result.model_dump(mode="json")
                _freeze_checkpoint(path, identity=identity, assessment=assessment)
                outputs[lens_type] = assessment
            finally:
                pending.task_done()

    await asyncio.gather(worker(1), worker(2))
    if set(outputs) != set(LensType):
        raise ValueError(f"Incomplete C2 context-free lens outputs for {bundle.case_id}")

    evaluation = DecisionLensEvaluation(
        decision_object_id=obj.decision_object_id,
        technical_feasibility=outputs[LensType.TECHNICAL_FEASIBILITY],
        institutional_regional=outputs[LensType.INSTITUTIONAL_REGIONAL],
        financial_adoption=outputs[LensType.FINANCIAL_ADOPTION],
    )
    recommendation, _ = _decision_recommendation(evaluation=evaluation)
    stances = {lens.value: int(outputs[lens]["stance"]) for lens in LensType}
    d_lens = round(float(pvariance(list(stances.values()))), 6)

    # One context-free DecisionObject is paired against all three contextual
    # jurisdiction scenarios. Replication is only for paired metric alignment;
    # it is not represented as three independent context-free generations.
    records: list[VariantResultRecord] = []
    for contextual in bundle.contextual_objects:
        scenario_id = str(contextual["scenario_id"])
        records.append(
            VariantResultRecord(
                axis="C2_CONTEXTUALIZED_PATH_COMPARISON",
                variant_id=CONTEXT_FREE_VARIANT_ID,
                case_id=bundle.case_id,
                scenario_id=scenario_id,
                decision_object_id=obj.decision_object_id,
                stance_by_lens=stances,
                d_lens=d_lens,
                evidence_ids_by_lens={lens.value: _evidence_ids(outputs[lens]) for lens in LensType},
                claim_directions_by_lens={lens.value: _directions(outputs[lens]) for lens in LensType},
                recommendation=recommendation.value,
                prompt_hashes={"base_stage4_prompt_set_sha256": sha256_json(prompt_hashes)},
                runtime=runtime,
                provenance={
                    "runner_version": STAGE7_DECISION_VARIANT_RUNNER_VERSION,
                    "context_free_single_generation_replicated_for_paired_scenario_alignment": True,
                    "base_stage3_artifact_sha256": stage3_artifact["artifact_sha256"],
                    "base_stage3_input_fingerprint": stage3_artifact["input_fingerprint"],
                    "base_stage3_output_sha256": stage3_artifact["stage3_output_sha256"],
                    "source_stage6_artifact_sha256": bundle.stage6["artifact_sha256"],
                },
            )
        )
    provenance = {
        "base_stage3_artifact_sha256": stage3_artifact["artifact_sha256"],
        "base_stage3_input_fingerprint": stage3_artifact["input_fingerprint"],
        "base_stage3_output_sha256": stage3_artifact["stage3_output_sha256"],
        "forecast_evidence_ids": sorted(obj.evaluation_evidence_ids),
        "context_or_decision_evidence_visible": False,
        "single_context_free_generation_per_case": True,
    }
    return records, provenance


async def run_contextualized_path_comparison(root: Path, bundles: list[CaseBundle]) -> list[Path]:
    records: list[VariantResultRecord] = []
    provenance_by_case: dict[str, Any] = {}
    for bundle in bundles:
        case_records, provenance = await _run_context_free_case(root, bundle=bundle)
        records.extend(case_records)
        provenance_by_case[bundle.case_id] = provenance
    return [
        _freeze_variant_records(
            root,
            axis="C2_CONTEXTUALIZED_PATH_COMPARISON",
            variant_id=CONTEXT_FREE_VARIANT_ID,
            repeat_index=None,
            records=records,
            provenance={
                "case_ids": [bundle.case_id for bundle in bundles],
                "by_case": provenance_by_case,
                "main_contextual_variant": "FULL_REDESIGNED_PIPELINE",
            },
        )
    ]


async def run_final_variants(
    root: Path,
    bundles: list[CaseBundle],
    *,
    axes: set[str] | None = None,
) -> list[Path]:
    requested = axes or {"A", "B", "C"}
    paths: list[Path] = []
    if "A" in requested:
        paths.extend(await run_prompt_paraphrase_variants(root, bundles))
    if "B" in requested:
        paths.append(await run_single_agent_baseline(root, bundles))
    if "C" in requested:
        paths.extend(await run_contextualized_path_comparison(root, bundles))
    return paths


__all__ = [
    "STAGE7_DECISION_VARIANT_RUNNER_VERSION",
    "run_final_variants",
    "run_prompt_paraphrase_variants",
    "run_contextualized_path_comparison",
    "run_single_agent_baseline",
]
