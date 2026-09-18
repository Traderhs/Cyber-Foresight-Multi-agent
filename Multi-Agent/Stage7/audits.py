from __future__ import annotations

import asyncio
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from langchain_core.messages import HumanMessage, SystemMessage

from Stage7.config import (
    MANUAL_AUDIT_PER_STRATUM,
    MANUAL_AUDIT_SELECTION_SEED,
    MEDIATOR_AUDIT_INCLUDE_ALL_FOLLOWUP_ROUNDS,
    MEDIATOR_AUDIT_PER_CASE_OUTCOME_STRATUM,
    MEDIATOR_AUDIT_SELECTION_SEED,
    STAGE7_GROUNDING_VERIFIER_VERSION,
    STAGE7_MANUAL_AUDIT_SAMPLE_VERSION,
    STAGE7_MEDIATOR_AUDIT_SAMPLE_VERSION,
    STAGE7_MEDIATOR_VERIFIER_VERSION,
)
from Stage7.loaders import CaseBundle, canonical_json, sha256_json, sha256_text
from Stage7.prompts import (
    GROUNDING_AUDIT_SYSTEM_PROMPT,
    GROUNDING_AUDIT_USER_PROMPT,
    MEDIATOR_AUDIT_SYSTEM_PROMPT,
    MEDIATOR_AUDIT_USER_PROMPT,
)
from Stage7.runtime import STAGE7_CLIENT_CONCURRENCY, get_stage7_llm, get_stage7_runtime_profile
from Stage7.schema import (
    GroundingAuditItem,
    GroundingAuditResponse,
    MediatorAuditItem,
    MediatorAuditResponse,
)


_INLINE_EVIDENCE_RE = re.compile(r"EVIDENCE:([A-Za-z0-9_.:-]+)")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9#*`])")


def _audit_id(*parts: str) -> str:
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _date_value(record: dict[str, Any]) -> str | None:
    for key in ("available_at", "evidence_date", "publication_date"):
        value = record.get(key)
        if value:
            return str(value)
    return None


def _deterministic_checks(
    *,
    evidence_ids: list[str],
    evidence_map: dict[str, dict[str, Any]],
    cutoff: str | None,
) -> dict[str, Any]:
    missing = sorted(set(evidence_ids) - set(evidence_map))
    records = [evidence_map[eid] for eid in evidence_ids if eid in evidence_map]
    temporal_violations: list[str] = []
    if cutoff:
        for record in records:
            value = _date_value(record)
            if value and value[:10] > cutoff[:10]:
                temporal_violations.append(str(record["evidence_id"]))
    missing_locator = [
        str(record["evidence_id"])
        for record in records
        if not str(record.get("source_locator") or "").strip()
    ]
    missing_content_hash = [
        str(record["evidence_id"])
        for record in records
        if not str(record.get("content_hash") or "").strip()
    ]
    return {
        "evidence_exists": not missing,
        "missing_evidence_ids": missing,
        "temporal_compliance": not temporal_violations,
        "temporal_violation_ids": temporal_violations,
        "source_locator_present": not missing_locator,
        "missing_source_locator_ids": missing_locator,
        "content_hash_present": not missing_content_hash,
        "missing_content_hash_ids": missing_content_hash,
    }


def _records(evidence_map: dict[str, dict[str, Any]], ids: Iterable[str]) -> list[dict[str, Any]]:
    return [evidence_map[eid] for eid in dict.fromkeys(ids) if eid in evidence_map]


def extract_stage1_grounding_items(bundle: CaseBundle) -> list[GroundingAuditItem]:
    evidence_map = bundle.evidence_records_by_id()
    cutoff = str(bundle.stage3_context.get("analysis_cutoff_date") or "") or None
    items: list[GroundingAuditItem] = []
    assessments = bundle.stage1.get("assessments") or {}
    for critic_name, assessment in assessments.items():
        claims = assessment.get("claims") or []
        opposite_by_relation: dict[str, list[str]] = {"SUPPORTS_FORECAST": [], "CHALLENGES_FORECAST": []}
        for claim in claims:
            relation = str(claim.get("forecast_relation") or "")
            if relation in opposite_by_relation:
                opposite_by_relation[relation].extend(claim.get("evidence_ids") or [])
        for claim in claims:
            evidence_ids = list(dict.fromkeys(claim.get("evidence_ids") or []))
            if not evidence_ids:
                continue
            relation = str(claim.get("forecast_relation") or "")
            opposite = (
                opposite_by_relation["CHALLENGES_FORECAST"]
                if relation == "SUPPORTS_FORECAST"
                else opposite_by_relation["SUPPORTS_FORECAST"]
                if relation == "CHALLENGES_FORECAST"
                else []
            )
            claim_id = str(claim.get("claim_id") or "")
            items.append(
                GroundingAuditItem(
                    audit_id=_audit_id("Stage1", bundle.case_id, critic_name, claim_id),
                    source_stage="Stage1",
                    case_id=bundle.case_id,
                    claim_id=claim_id,
                    claim_relation=relation or None,
                    statement=str(claim.get("statement") or ""),
                    evidence_ids=evidence_ids,
                    evidence_records=_records(evidence_map, evidence_ids),
                    counter_evidence_records=_records(evidence_map, opposite)[:8],
                    analysis_cutoff_date=cutoff,
                    deterministic_checks=_deterministic_checks(
                        evidence_ids=evidence_ids, evidence_map=evidence_map, cutoff=cutoff
                    ),
                )
            )
    return items


def extract_stage4_grounding_items(bundle: CaseBundle) -> list[GroundingAuditItem]:
    evidence_map = bundle.evidence_records_by_id()
    cutoff = str(bundle.stage3_context.get("analysis_cutoff_date") or "") or None
    items: list[GroundingAuditItem] = []
    decision_to_scenario = {
        item["decision_object_id"]: item["scenario_id"] for item in bundle.syntheses
    }
    for evaluation in bundle.stage4_evaluations:
        decision_object_id = str(evaluation["decision_object_id"])
        scenario_id = str(decision_to_scenario[decision_object_id])
        for lens in ("technical_feasibility", "institutional_regional", "financial_adoption"):
            assessment = evaluation[lens]
            claims = assessment.get("claims") or []
            support_ids = [
                eid
                for claim in claims
                if claim.get("status") == "SUPPORTS_FEASIBILITY"
                for eid in claim.get("evidence_ids") or []
            ]
            challenge_ids = [
                eid
                for claim in claims
                if claim.get("status") == "CHALLENGES_FEASIBILITY"
                for eid in claim.get("evidence_ids") or []
            ]
            for claim in claims:
                evidence_ids = list(dict.fromkeys(claim.get("evidence_ids") or []))
                if not evidence_ids:
                    continue
                status = str(claim.get("status") or "")
                opposite = (
                    challenge_ids
                    if status == "SUPPORTS_FEASIBILITY"
                    else support_ids
                    if status == "CHALLENGES_FEASIBILITY"
                    else [*support_ids, *challenge_ids]
                )
                claim_id = str(claim.get("claim_id") or "")
                items.append(
                    GroundingAuditItem(
                        audit_id=_audit_id(
                            "Stage4", bundle.case_id, scenario_id, lens, claim_id
                        ),
                        source_stage="Stage4",
                        case_id=bundle.case_id,
                        scenario_id=scenario_id,
                        lens=lens,
                        claim_id=claim_id,
                        claim_relation=status or None,
                        statement=str(claim.get("statement") or ""),
                        evidence_ids=evidence_ids,
                        evidence_records=_records(evidence_map, evidence_ids),
                        counter_evidence_records=_records(evidence_map, opposite)[:8],
                        analysis_cutoff_date=cutoff,
                        deterministic_checks=_deterministic_checks(
                            evidence_ids=evidence_ids, evidence_map=evidence_map, cutoff=cutoff
                        ),
                    )
                )
    return items


def _report_claim_chunks(report: str) -> list[tuple[str, list[str]]]:
    chunks: list[tuple[str, list[str]]] = []
    for paragraph in re.split(r"\n\s*\n|\n(?=[-*#])", report):
        paragraph = paragraph.strip()
        if not paragraph or "EVIDENCE:" not in paragraph:
            continue
        pieces = _SENTENCE_SPLIT_RE.split(paragraph)
        for piece in pieces:
            evidence_ids = list(dict.fromkeys(_INLINE_EVIDENCE_RE.findall(piece)))
            if not evidence_ids:
                continue
            statement = re.sub(r"\[EVIDENCE:[^\]]+\]", "", piece).strip(" -*#\t")
            if statement:
                chunks.append((statement, evidence_ids))
    return chunks


def extract_stage6_grounding_items(bundle: CaseBundle) -> list[GroundingAuditItem]:
    evidence_map = bundle.evidence_records_by_id()
    cutoff = str(bundle.stage3_context.get("analysis_cutoff_date") or "") or None
    items: list[GroundingAuditItem] = []
    for synthesis in bundle.syntheses:
        scenario_id = str(synthesis["scenario_id"])
        lens_counter_ids: list[str] = []
        for lens_payload in (synthesis.get("lens_evidence_trace") or {}).values():
            for claim in lens_payload.get("claims") or []:
                lens_counter_ids.extend(claim.get("evidence_ids") or [])
        for index, (statement, evidence_ids) in enumerate(
            _report_claim_chunks(str(synthesis.get("strategic_intelligence_report") or "")),
            start=1,
        ):
            counter_ids = [eid for eid in dict.fromkeys(lens_counter_ids) if eid not in evidence_ids]
            items.append(
                GroundingAuditItem(
                    audit_id=_audit_id("Stage6", bundle.case_id, scenario_id, str(index), statement),
                    source_stage="Stage6",
                    case_id=bundle.case_id,
                    scenario_id=scenario_id,
                    claim_id=f"REPORT_SENT_{index:03d}",
                    statement=statement,
                    evidence_ids=evidence_ids,
                    evidence_records=_records(evidence_map, evidence_ids),
                    counter_evidence_records=_records(evidence_map, counter_ids)[:8],
                    analysis_cutoff_date=cutoff,
                    deterministic_checks=_deterministic_checks(
                        evidence_ids=evidence_ids, evidence_map=evidence_map, cutoff=cutoff
                    ),
                )
            )
    return items


def extract_all_grounding_items(bundles: list[CaseBundle]) -> list[GroundingAuditItem]:
    items: list[GroundingAuditItem] = []
    for bundle in bundles:
        items.extend(extract_stage1_grounding_items(bundle))
        items.extend(extract_stage4_grounding_items(bundle))
        items.extend(extract_stage6_grounding_items(bundle))
    return items


def extract_mediator_items(bundle: CaseBundle) -> list[MediatorAuditItem]:
    result = bundle.stage2_result
    evidence_map = bundle.evidence_records_by_id()
    attack_claims = {
        str(claim["claim_id"]): claim
        for claim in (result.get("attack_pre_assessment") or {}).get("claims") or []
    }
    defense_claims = {
        str(claim["claim_id"]): claim
        for claim in (result.get("defense_pre_assessment") or {}).get("claims") or []
    }
    items: list[MediatorAuditItem] = []
    for round_payload in result.get("rounds") or []:
        mediator = round_payload.get("mediator_summary") or {}
        round_index = int(round_payload.get("round_index") or mediator.get("round_index") or 0)
        next_focus = mediator.get("next_round_focus") or []
        for adjudication in mediator.get("adjudications") or []:
            evidence_ids = list(dict.fromkeys(adjudication.get("evidence_ids") or []))
            adjudication_id = str(adjudication.get("adjudication_id") or "")
            items.append(
                MediatorAuditItem(
                    audit_id=_audit_id(
                        "Stage2", bundle.case_id, str(round_index), adjudication_id
                    ),
                    case_id=bundle.case_id,
                    round_index=round_index,
                    adjudication_id=adjudication_id,
                    outcome=str(adjudication.get("outcome") or ""),
                    attack_claims=[
                        attack_claims[claim_id]
                        for claim_id in adjudication.get("attack_claim_ids") or []
                        if claim_id in attack_claims
                    ],
                    defense_claims=[
                        defense_claims[claim_id]
                        for claim_id in adjudication.get("defense_claim_ids") or []
                        if claim_id in defense_claims
                    ],
                    cited_evidence_records=_records(evidence_map, evidence_ids),
                    rationale=str(adjudication.get("rationale") or ""),
                    required_revision=adjudication.get("required_revision"),
                    round_action=str(mediator.get("round_action") or ""),
                    next_round_focus=list(next_focus),
                )
            )
    return items


def extract_all_mediator_items(bundles: list[CaseBundle]) -> list[MediatorAuditItem]:
    return [item for bundle in bundles for item in extract_mediator_items(bundle)]


def _sample_score(seed: int, audit_id: str) -> str:
    return hashlib.sha256(f"{seed}|{audit_id}".encode("utf-8")).hexdigest()


def select_manual_audit_sample(
    items: list[GroundingAuditItem],
    *,
    seed: int = MANUAL_AUDIT_SELECTION_SEED,
    per_stratum: int = MANUAL_AUDIT_PER_STRATUM,
) -> list[GroundingAuditItem]:
    strata: dict[tuple[str, str, str], list[GroundingAuditItem]] = {}
    for item in items:
        key = (item.source_stage, item.case_id, item.lens or "NO_LENS")
        strata.setdefault(key, []).append(item)
    selected: list[GroundingAuditItem] = []
    for key in sorted(strata):
        candidates = sorted(strata[key], key=lambda item: _sample_score(seed, item.audit_id))
        selected.extend(candidates[:per_stratum])
    return selected


def select_mediator_audit_sample(
    items: list[MediatorAuditItem],
    *,
    seed: int = MEDIATOR_AUDIT_SELECTION_SEED,
    per_stratum: int = MEDIATOR_AUDIT_PER_CASE_OUTCOME_STRATUM,
    include_all_followup_rounds: bool = MEDIATOR_AUDIT_INCLUDE_ALL_FOLLOWUP_ROUNDS,
) -> list[MediatorAuditItem]:
    """Predeclared representative semantic-audit subset.

    One deterministic item is selected per (case, adjudication outcome) stratum,
    and every follow-up-round adjudication is retained because Round 2 is rare
    and directly tests the bounded-continuation contract. Selection never uses
    verifier outputs or Stage 6 recommendations.
    """

    strata: dict[tuple[str, str], list[MediatorAuditItem]] = {}
    for item in items:
        strata.setdefault((item.case_id, item.outcome), []).append(item)

    selected: dict[str, MediatorAuditItem] = {}
    for key in sorted(strata):
        candidates = sorted(
            strata[key], key=lambda item: _sample_score(seed, item.audit_id)
        )
        for item in candidates[:per_stratum]:
            selected[item.audit_id] = item

    if include_all_followup_rounds:
        for item in items:
            if item.round_index > 1:
                selected[item.audit_id] = item

    return [selected[audit_id] for audit_id in sorted(selected)]


def write_mediator_audit_sample(root: Path, items: list[MediatorAuditItem]) -> Path:
    path = (
        root
        / "Multi-Agent"
        / "Results"
        / "Stage7"
        / "manual_audit"
        / f"{STAGE7_MEDIATOR_AUDIT_SAMPLE_VERSION}.json"
    )
    selected = select_mediator_audit_sample(items)
    payload = {
        "schema_version": STAGE7_MEDIATOR_AUDIT_SAMPLE_VERSION,
        "selection_seed": MEDIATOR_AUDIT_SELECTION_SEED,
        "per_case_outcome_stratum": MEDIATOR_AUDIT_PER_CASE_OUTCOME_STRATUM,
        "include_all_followup_rounds": MEDIATOR_AUDIT_INCLUDE_ALL_FOLLOWUP_ROUNDS,
        "population_count": len(items),
        "sample_count": len(selected),
        "selection_rule": (
            "one deterministic hash-selected adjudication per (case_id,outcome) stratum; "
            "plus every round_index>1 adjudication; no verifier output or Stage 6 recommendation used"
        ),
        "items": [item.model_dump(mode="json") for item in selected],
    }
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != serialized:
        raise ValueError(
            f"Frozen Stage 7 Mediator-audit sample changed in place: {path}. "
            "Bump the sample version."
        )
    if not path.exists():
        path.write_text(serialized, encoding="utf-8", newline="\n")
    return path


def write_manual_audit_sample(root: Path, items: list[GroundingAuditItem]) -> Path:
    path = (
        root
        / "Multi-Agent"
        / "Results"
        / "Stage7"
        / "manual_audit"
        / f"{STAGE7_MANUAL_AUDIT_SAMPLE_VERSION}.json"
    )
    payload = {
        "schema_version": STAGE7_MANUAL_AUDIT_SAMPLE_VERSION,
        "selection_seed": MANUAL_AUDIT_SELECTION_SEED,
        "per_stratum": MANUAL_AUDIT_PER_STRATUM,
        "items": [item.model_dump(mode="json") for item in select_manual_audit_sample(items)],
    }
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != serialized:
        raise ValueError(
            f"Frozen Stage 7 manual-audit sample changed in place: {path}. Bump the sample version."
        )
    if not path.exists():
        path.write_text(serialized, encoding="utf-8", newline="\n")
    return path


def _checkpoint_identity(kind: str, item_payload: dict[str, Any]) -> dict[str, Any]:
    if kind == "grounding":
        verifier_version = STAGE7_GROUNDING_VERIFIER_VERSION
        system_prompt = GROUNDING_AUDIT_SYSTEM_PROMPT
        user_prompt = GROUNDING_AUDIT_USER_PROMPT
        output_schema = GroundingAuditResponse.model_json_schema()
    elif kind == "mediator":
        verifier_version = STAGE7_MEDIATOR_VERIFIER_VERSION
        system_prompt = MEDIATOR_AUDIT_SYSTEM_PROMPT
        user_prompt = MEDIATOR_AUDIT_USER_PROMPT
        output_schema = MediatorAuditResponse.model_json_schema()
    else:
        raise ValueError(f"Unknown audit kind: {kind}")
    identity = {
        "kind": kind,
        "verifier_version": verifier_version,
        "item_sha256": sha256_json(item_payload),
        "system_prompt_sha256": sha256_text(system_prompt),
        "user_prompt_sha256": sha256_text(user_prompt),
        "output_schema_sha256": sha256_json(output_schema),
        "runtime": get_stage7_runtime_profile(),
    }
    identity["input_fingerprint"] = sha256_json(identity)
    return identity


def _checkpoint_path(root: Path, kind: str, item_id: str, identity: dict[str, Any]) -> Path:
    safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", item_id)
    return (
        root
        / "Multi-Agent"
        / "Results"
        / "Stage7"
        / "checkpoints"
        / kind
        / identity["input_fingerprint"]
        / f"{safe_id}.json"
    )


def _load_checkpoint(path: Path, identity: dict[str, Any]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    stored_hash = str(payload.get("checkpoint_sha256") or "")
    unhashed = {key: value for key, value in payload.items() if key != "checkpoint_sha256"}
    if not stored_hash or stored_hash != sha256_json(unhashed):
        raise ValueError(f"Stage 7 checkpoint hash mismatch: {path}")
    if payload.get("identity") != identity:
        raise ValueError(f"Stage 7 checkpoint identity mismatch: {path}")
    return payload


def _freeze_checkpoint(
    path: Path,
    *,
    identity: dict[str, Any],
    item_id: str,
    response: dict[str, Any],
) -> None:
    payload = {
        "identity": identity,
        "item_id": item_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "response": response,
        "response_sha256": sha256_json(response),
    }
    payload["checkpoint_sha256"] = sha256_json(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        existing = _load_checkpoint(path, identity)
        if existing is None or existing.get("response_sha256") != payload["response_sha256"]:
            raise ValueError(f"Conflicting Stage 7 checkpoint: {path}")
        return
    path.write_text(serialized, encoding="utf-8", newline="\n")


def load_cached_audit_responses(
    root: Path,
    *,
    kind: str,
    items: list[GroundingAuditItem] | list[MediatorAuditItem],
) -> list[dict[str, Any]]:
    """Return a complete cached audit set, or an empty list if any item is missing.

    Every checkpoint is revalidated against the current verifier prompt/schema/runtime
    identity before reuse. Partial cached audit sets are never mixed into a frozen
    Stage 7 artifact.
    """

    responses: list[dict[str, Any]] = []
    for item in items:
        payload = item.model_dump(mode="json")
        identity = _checkpoint_identity(kind, payload)
        path = _checkpoint_path(root, kind, item.audit_id, identity)
        checkpoint = _load_checkpoint(path, identity)
        if checkpoint is None:
            return []
        responses.append(dict(checkpoint["response"]))
    return responses


async def run_grounding_verifier(
    root: Path,
    items: list[GroundingAuditItem],
) -> list[dict[str, Any]]:
    client = get_stage7_llm()
    outputs: dict[str, dict[str, Any]] = {}
    pending: asyncio.Queue[GroundingAuditItem] = asyncio.Queue()
    identities: dict[str, dict[str, Any]] = {}
    paths: dict[str, Path] = {}

    for item in items:
        payload = item.model_dump(mode="json")
        identity = _checkpoint_identity("grounding", payload)
        path = _checkpoint_path(root, "grounding", item.audit_id, identity)
        identities[item.audit_id] = identity
        paths[item.audit_id] = path
        checkpoint = _load_checkpoint(path, identity)
        if checkpoint is not None:
            outputs[item.audit_id] = dict(checkpoint["response"])
        else:
            pending.put_nowait(item)

    print(
        f"Stage 7 grounding queue: {pending.qsize()} missing of {len(items)} audits; "
        f"{STAGE7_CLIENT_CONCURRENCY} workers share the existing two-slot llama.cpp runtime.",
        flush=True,
    )

    async def worker(worker_index: int) -> None:
        while True:
            try:
                item = pending.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                runner = client.with_structured_output(
                    GroundingAuditResponse,
                    method="json_schema",
                    progress_label=f"Stage7-Grounding-W{worker_index}-{item.audit_id[:10]}",
                )
                response = await runner.ainvoke(
                    [
                        SystemMessage(content=GROUNDING_AUDIT_SYSTEM_PROMPT),
                        HumanMessage(
                            content=GROUNDING_AUDIT_USER_PROMPT.format(
                                audit_item=json.dumps(
                                    item.model_dump(mode="json"),
                                    ensure_ascii=False,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                )
                            )
                        ),
                    ]
                )
                # audit_id is deterministic envelope metadata, not a semantic
                # judgment the verifier is allowed to mutate.  Some constrained
                # generations can reproduce a long hash-like ID with a one-char
                # transcription error, so always bind it back to the frozen item
                # before validation/checkpointing.
                response = response.model_copy(update={"audit_id": item.audit_id})
                allowed = set(item.evidence_ids) | {
                    str(record.get("evidence_id"))
                    for record in item.counter_evidence_records
                    if record.get("evidence_id")
                }
                if set(response.decisive_evidence_ids) - allowed:
                    raise ValueError("Grounding verifier cited evidence outside supplied audit boundary")
                payload = response.model_dump(mode="json")
                _freeze_checkpoint(
                    paths[item.audit_id],
                    identity=identities[item.audit_id],
                    item_id=item.audit_id,
                    response=payload,
                )
                outputs[item.audit_id] = payload
            finally:
                pending.task_done()

    await asyncio.gather(
        *[worker(index + 1) for index in range(STAGE7_CLIENT_CONCURRENCY)]
    )
    return [outputs[item.audit_id] for item in items]


async def run_mediator_verifier(
    root: Path,
    items: list[MediatorAuditItem],
) -> list[dict[str, Any]]:
    client = get_stage7_llm()
    outputs: dict[str, dict[str, Any]] = {}
    pending: asyncio.Queue[MediatorAuditItem] = asyncio.Queue()
    identities: dict[str, dict[str, Any]] = {}
    paths: dict[str, Path] = {}

    for item in items:
        payload = item.model_dump(mode="json")
        identity = _checkpoint_identity("mediator", payload)
        path = _checkpoint_path(root, "mediator", item.audit_id, identity)
        identities[item.audit_id] = identity
        paths[item.audit_id] = path
        checkpoint = _load_checkpoint(path, identity)
        if checkpoint is not None:
            outputs[item.audit_id] = dict(checkpoint["response"])
        else:
            pending.put_nowait(item)

    print(
        f"Stage 7 mediator queue: {pending.qsize()} missing of {len(items)} audits; "
        f"{STAGE7_CLIENT_CONCURRENCY} workers share the existing two-slot llama.cpp runtime.",
        flush=True,
    )

    async def worker(worker_index: int) -> None:
        while True:
            try:
                item = pending.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                runner = client.with_structured_output(
                    MediatorAuditResponse,
                    method="json_schema",
                    progress_label=f"Stage7-Mediator-W{worker_index}-{item.audit_id[:10]}",
                )
                response = await runner.ainvoke(
                    [
                        SystemMessage(content=MEDIATOR_AUDIT_SYSTEM_PROMPT),
                        HumanMessage(
                            content=MEDIATOR_AUDIT_USER_PROMPT.format(
                                audit_item=json.dumps(
                                    item.model_dump(mode="json"),
                                    ensure_ascii=False,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                )
                            )
                        ),
                    ]
                )
                response = response.model_copy(update={"audit_id": item.audit_id})
                payload = response.model_dump(mode="json")
                _freeze_checkpoint(
                    paths[item.audit_id],
                    identity=identities[item.audit_id],
                    item_id=item.audit_id,
                    response=payload,
                )
                outputs[item.audit_id] = payload
            finally:
                pending.task_done()

    await asyncio.gather(
        *[worker(index + 1) for index in range(STAGE7_CLIENT_CONCURRENCY)]
    )
    return [outputs[item.audit_id] for item in items]
