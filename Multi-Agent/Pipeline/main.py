from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import json
import os
import random
import sys
import threading
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, TextIO

from dotenv import load_dotenv

from Pipeline.graph import create_graph
from Stage0.agent_input import build_agent_input
from Stage0.action_evidence import (
    ACTION_EVIDENCE_REGISTRY_SHA256,
    ACTION_EVIDENCE_REGISTRY_VERSION,
    ACTION_EVIDENCE_SNAPSHOT_ID,
    load_action_evidence_records,
)
from Stage0.context_scenarios import CONTEXT_SCENARIO_SET_VERSION, main_context_scenarios
from Stage0.decision_evidence import (
    DECISION_EVIDENCE_RETRIEVAL_VERSION,
    MAIN_MIN_BALANCE_DOCUMENTS_PER_FACET,
    MAIN_MIN_BALANCE_PUBLISHERS_PER_FACET,
    MAIN_MIN_GENERIC_DECISION_EVIDENCE_DOCUMENTS,
    MAIN_MIN_GENERIC_DECISION_EVIDENCE_RECORDS,
    MAIN_MIN_LENS_DECISION_EVIDENCE_DOCUMENTS,
    MAIN_MIN_LENS_DECISION_EVIDENCE_PUBLISHERS,
    build_decision_evidence_pack,
    open_snapshot_for_decision_evidence,
)
from Stage0.decision_sources import (
    DECISION_SOURCE_RECORDS_SHA256,
    DECISION_SOURCE_REGISTRY_VERSION,
    DECISION_SOURCE_SNAPSHOT_ID,
    DECISION_SOURCE_SPEC_MANIFEST_SHA256,
    load_decision_source_snapshot,
)
from Stage0.main_guidance_selection import (
    MAIN_GUIDANCE_SELECTION_SHA256,
    MAIN_GUIDANCE_SELECTION_VERSION,
    load_main_guidance_selection,
)
from Stage1.artifact import STAGE1_ARTIFACT_SCHEMA_VERSION
from Stage1.cases import (
    MAIN_ANALYSIS_CUTOFF,
    MAIN_CASE_SET_VERSION,
    MAIN_EVALUATION_MODE,
    MAIN_SNAPSHOT_ID,
    MAIN_STAGE1_CASES,
    REGIME_EVIDENCE_REQUIREMENTS,
    MainStage1Case,
    main_case_set_manifest,
    validate_main_case_set,
)
from Stage1.nodes import _run_stage1_critic
from Stage1.prompts import ATTACK_FEASIBILITY_SYSTEM_PROMPT, DEFENSE_ROBUSTNESS_SYSTEM_PROMPT
from Stage1.runtime import get_runtime_config
from Stage1.schema import CriticType
from Stage4.context_artifact import (
    STAGE4_CONTEXT_ARTIFACT_SCHEMA_VERSION,
    STAGE4_CONTEXT_GENERATION_CONTRACT_VERSION,
)
from Stage4.context_runtime import (
    STAGE4_CONTEXT_RUNTIME_PROFILE_VERSION,
    STAGE4_CONTEXT_SIZE,
    STAGE4_PARALLEL_SLOTS,
    STAGE4_SERVER_CONTEXT_SIZE,
)
from Stage4.context_schema import STAGE4_CONTEXT_SEMANTIC_VALIDATION_VERSION
from Stage5.artifact import STAGE5_ARTIFACT_SCHEMA_VERSION
from Stage5.schema import STAGE5_DIAGNOSTIC_CONTRACT_VERSION
from Stage6.artifact import (
    STAGE6_ARTIFACT_SCHEMA_VERSION,
    STAGE6_GENERATION_CONTRACT_VERSION,
)
from Stage6.runtime import (
    STAGE6_CONTEXT_SIZE,
    STAGE6_PARALLEL_SLOTS,
    STAGE6_RUNTIME_PROFILE_VERSION,
    STAGE6_SERVER_CONTEXT_SIZE,
)
from Stage6.schema import (
    STAGE6_DECISION_POLICY_VERSION,
    STAGE6_SEMANTIC_VALIDATION_VERSION,
    STAGE6_SYNTHESIS_CONTRACT_VERSION,
)


load_dotenv()


class _TeeTextIO:
    """Line-buffered console + file sink so interrupted runs keep their logs."""

    def __init__(self, terminal: TextIO, log_file: TextIO):
        self.terminal = terminal
        self.log_file = log_file
        self._lock = threading.Lock()

    def write(self, text: str) -> int:
        with self._lock:
            # The console/PTY is observational only. DevSpace or an attached
            # terminal can disappear while a long-running main experiment is
            # still healthy. Never let a broken console handle abort inference
            # or artifact creation; the file log remains the durable sink.
            try:
                self.terminal.write(text)
                self.terminal.flush()
            except (OSError, ValueError):
                pass
            self.log_file.write(text)
            self.log_file.flush()
        return len(text)

    def flush(self) -> None:
        with self._lock:
            try:
                self.terminal.flush()
            except (OSError, ValueError):
                pass
            self.log_file.flush()

    @property
    def encoding(self) -> str | None:
        return getattr(self.terminal, "encoding", "utf-8")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(path)


def _ensure_frozen_case_set_manifest(root: Path) -> Path:
    manifest = main_case_set_manifest()
    path = root / "Multi-Agent" / "Results" / "Stage1" / "case_sets" / f"{MAIN_CASE_SET_VERSION}.json"
    serialized = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing != serialized:
            raise RuntimeError(
                f"Frozen main case-set manifest changed in place: {path}. "
                "Create a new MAIN_CASE_SET_VERSION instead of overwriting an experiment definition."
            )
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(serialized, encoding="utf-8", newline="\n")
    return path


def _initial_state(stage0_config: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "stage0_config": stage0_config,
        "forecast_data": "",
        "evidence_pack": None,
        "attack_assessment": None,
        "defense_assessment": None,
        "stage1_complete": False,
        "stage1_artifact_reused": False,
        "attack_checkpoint_reused": False,
        "defense_checkpoint_reused": False,
        "stage1_input_fingerprint": None,
        "stage1_artifact_path": None,
        "stage2_complete": False,
        "stage2_artifact_reused": False,
        "stage2_input_fingerprint": None,
        "stage2_artifact_path": None,
        "stage2_debate_result": None,
        "debate_rounds": [],
        "attack_post_assessment": None,
        "defense_post_assessment": None,
        "stage3_complete": False,
        "stage3_artifact_reused": False,
        "stage3_input_fingerprint": None,
        "stage3_artifact_path": None,
        "stage3_decision_bundle": None,
        "decision_objects": [],
        "stage3_context_complete": False,
        "stage3_context_artifact_reused": False,
        "stage3_context_input_fingerprint": None,
        "stage3_context_artifact_path": None,
        "stage3_contextual_decision_bundle": None,
        "contextual_decision_objects": [],
        "stage4_complete": False,
        "stage4_artifact_reused": False,
        "stage4_input_fingerprint": None,
        "stage4_artifact_path": None,
        "stage4_evaluation_bundle": None,
        "technical_lens_assessments": [],
        "institutional_lens_assessments": [],
        "financial_lens_assessments": [],
        "technical_checkpoint_reuse_count": 0,
        "institutional_checkpoint_reuse_count": 0,
        "financial_checkpoint_reuse_count": 0,
        "stage4_context_complete": False,
        "stage4_context_artifact_reused": False,
        "stage4_context_input_fingerprint": None,
        "stage4_context_artifact_path": None,
        "stage4_context_evaluation_bundle": None,
        "contextual_technical_lens_assessments": [],
        "contextual_institutional_lens_assessments": [],
        "contextual_financial_lens_assessments": [],
        "contextual_technical_checkpoint_reuse_count": 0,
        "contextual_institutional_checkpoint_reuse_count": 0,
        "contextual_financial_checkpoint_reuse_count": 0,
        "stage5_complete": False,
        "stage5_artifact_reused": False,
        "stage5_input_fingerprint": None,
        "stage5_artifact_path": None,
        "stage5_diagnostic_bundle": None,
        "stage6_complete": False,
        "stage6_artifact_reused": False,
        "stage6_input_fingerprint": None,
        "stage6_artifact_path": None,
        "stage6_synthesis_bundle": None,
        "decision_syntheses": [],
        "stage6_narratives": [],
        "stage6_checkpoint_reuse_count": 0,
        "messages": [],
    }


def _case_config(case: MainStage1Case) -> dict[str, Any]:
    config = {
        "snapshot_id": MAIN_SNAPSHOT_ID,
        "threat": case.threat,
        "pmt": case.pmt,
        "analysis_cutoff_date": MAIN_ANALYSIS_CUTOFF,
        "evaluation_mode": MAIN_EVALUATION_MODE,
        "case_id": case.case_id,
    }
    if case.selection_class == "regime_representative":
        config["min_covered_evidence_slots"] = REGIME_EVIDENCE_REQUIREMENTS[
            "min_covered_evidence_slots"
        ]
        config["min_source_families"] = REGIME_EVIDENCE_REQUIREMENTS["min_source_families"]
    return config


def _decision_contract_tag() -> str:
    return hashlib.sha256(
        (
            f"{DECISION_EVIDENCE_RETRIEVAL_VERSION}|"
            f"{DECISION_SOURCE_REGISTRY_VERSION}|{DECISION_SOURCE_SNAPSHOT_ID}|"
            f"{DECISION_SOURCE_SPEC_MANIFEST_SHA256}|{DECISION_SOURCE_RECORDS_SHA256}|"
            f"{ACTION_EVIDENCE_REGISTRY_VERSION}|{ACTION_EVIDENCE_SNAPSHOT_ID}|"
            f"{ACTION_EVIDENCE_REGISTRY_SHA256}|"
            f"{MAIN_GUIDANCE_SELECTION_VERSION}|{MAIN_GUIDANCE_SELECTION_SHA256}"
        ).encode("utf-8")
    ).hexdigest()[:10]


def _preflight_main_decision_evidence(root: Path) -> Path:
    """Fail the whole main run before inference if any case lacks Stage-4 evidence depth."""

    store = open_snapshot_for_decision_evidence(project_root=root, snapshot_id=MAIN_SNAPSHOT_ID)
    supplemental_records, supplemental_manifest = load_decision_source_snapshot(project_root=root)
    supplemental_records_tuple = tuple(supplemental_records)
    action_records, action_manifest = load_action_evidence_records()
    action_records_tuple = tuple(action_records)
    if supplemental_manifest.get("manifest_sha256") != DECISION_SOURCE_SPEC_MANIFEST_SHA256:
        raise RuntimeError("Decision-source preflight rejected the frozen source-spec manifest hash")
    if supplemental_manifest.get("records_sha256") != DECISION_SOURCE_RECORDS_SHA256:
        raise RuntimeError("Decision-source preflight rejected the frozen records hash")

    rows: list[dict[str, Any]] = []
    for case in MAIN_STAGE1_CASES:
        stage0 = build_agent_input(
            project_root=root,
            snapshot_id=MAIN_SNAPSHOT_ID,
            threat=case.threat,
            pmt=case.pmt,
            analysis_cutoff_date=MAIN_ANALYSIS_CUTOFF,
            evaluation_mode=MAIN_EVALUATION_MODE,
            case_id=case.case_id,
        )
        pack = stage0["evidence_pack"]
        main_guidance_records, main_guidance_slot_coverage, main_guidance_manifest = (
            load_main_guidance_selection(
                pmt_id=str(pack["pmt_id"]),
                records=(*store.records, *supplemental_records_tuple),
            )
        )
        main_guidance_records_tuple = tuple(main_guidance_records)
        generic_sets: list[set[str]] = []
        scenario_rows: list[dict[str, Any]] = []
        for scenario in main_context_scenarios():
            decision_pack = build_decision_evidence_pack(
                store=store,
                cutoff_date=MAIN_ANALYSIS_CUTOFF,
                threat_id=str(pack["threat_id"]),
                pmt_id=str(pack["pmt_id"]),
                threat_name=case.threat,
                pmt_name=case.pmt,
                scenario=scenario,
                supplemental_records=supplemental_records_tuple,
                supplemental_manifest=supplemental_manifest,
                action_records=action_records_tuple,
                action_manifest=action_manifest,
                main_guidance_records=main_guidance_records_tuple,
                main_guidance_slot_coverage=main_guidance_slot_coverage,
                main_guidance_manifest=main_guidance_manifest,
                minimum_generic_records=MAIN_MIN_GENERIC_DECISION_EVIDENCE_RECORDS,
                minimum_generic_documents=MAIN_MIN_GENERIC_DECISION_EVIDENCE_DOCUMENTS,
                minimum_lens_documents=MAIN_MIN_LENS_DECISION_EVIDENCE_DOCUMENTS,
                minimum_balance_documents_per_facet=MAIN_MIN_BALANCE_DOCUMENTS_PER_FACET,
                minimum_balance_publishers_per_facet=MAIN_MIN_BALANCE_PUBLISHERS_PER_FACET,
                minimum_lens_publishers=MAIN_MIN_LENS_DECISION_EVIDENCE_PUBLISHERS,
            )
            coverage_floor = decision_pack["retrieval_metadata"]["coverage_floor"]
            regulatory_ids = {
                evidence_id
                for evidence_id, slots in decision_pack["slot_coverage"].items()
                if "regulatory_applicability" in slots
            }
            generic_sets.append(set(decision_pack["decision_evidence_ids"]) - regulatory_ids)
            scenario_rows.append(
                {
                    "scenario_id": scenario["scenario_id"],
                    "generic_record_count": coverage_floor["generic_record_count"],
                    "generic_document_count": coverage_floor["generic_document_count"],
                    "generic_document_ids": coverage_floor["generic_document_ids"],
                    "lens_coverage": coverage_floor["lens_coverage"],
                    "covered_slots": decision_pack["retrieval_metadata"]["covered_slots"],
                }
            )
        if any(item != generic_sets[0] for item in generic_sets[1:]):
            raise RuntimeError(
                f"Decision-evidence preflight failed: generic evidence changed by jurisdiction for {case.case_id}"
            )
        rows.append(
            {
                "case_id": case.case_id,
                "threat": case.threat,
                "pmt": case.pmt,
                "scenarios": scenario_rows,
            }
        )

    report = {
        "preflight_version": "stage3-decision-evidence-preflight-v3",
        "case_set_version": MAIN_CASE_SET_VERSION,
        "source_snapshot_id": MAIN_SNAPSHOT_ID,
        "analysis_cutoff_date": MAIN_ANALYSIS_CUTOFF,
        "decision_evidence_retrieval_version": DECISION_EVIDENCE_RETRIEVAL_VERSION,
        "decision_source_registry_version": DECISION_SOURCE_REGISTRY_VERSION,
        "decision_source_snapshot_id": DECISION_SOURCE_SNAPSHOT_ID,
        "decision_source_manifest_sha256": DECISION_SOURCE_SPEC_MANIFEST_SHA256,
        "decision_source_records_sha256": DECISION_SOURCE_RECORDS_SHA256,
        "action_evidence_registry_version": ACTION_EVIDENCE_REGISTRY_VERSION,
        "action_evidence_snapshot_id": ACTION_EVIDENCE_SNAPSHOT_ID,
        "action_evidence_registry_sha256": ACTION_EVIDENCE_REGISTRY_SHA256,
        "main_guidance_selection_version": MAIN_GUIDANCE_SELECTION_VERSION,
        "main_guidance_selection_sha256": MAIN_GUIDANCE_SELECTION_SHA256,
        "minimum_generic_records": MAIN_MIN_GENERIC_DECISION_EVIDENCE_RECORDS,
        "minimum_generic_documents": MAIN_MIN_GENERIC_DECISION_EVIDENCE_DOCUMENTS,
        "minimum_lens_documents": MAIN_MIN_LENS_DECISION_EVIDENCE_DOCUMENTS,
        "minimum_balance_documents_per_facet": MAIN_MIN_BALANCE_DOCUMENTS_PER_FACET,
        "minimum_balance_publishers_per_facet": MAIN_MIN_BALANCE_PUBLISHERS_PER_FACET,
        "minimum_lens_publishers": MAIN_MIN_LENS_DECISION_EVIDENCE_PUBLISHERS,
        "case_count": len(rows),
        "scenario_count": sum(len(row["scenarios"]) for row in rows),
        "status": "PASS",
        "cases": rows,
    }
    path = (
        root
        / "Multi-Agent"
        / "Results"
        / "Stage3"
        / "preflight"
        / f"{MAIN_CASE_SET_VERSION}__de-{_decision_contract_tag()}.json"
    )
    _write_json_atomic(path, report)
    return path


def _progress_path(root: Path, seed: int) -> Path:
    return (
        root
        / "Multi-Agent"
        / "Results"
        / "Stage6"
        / "runs"
        / (
            f"{MAIN_CASE_SET_VERSION}__{CONTEXT_SCENARIO_SET_VERSION}__"
            f"{STAGE4_CONTEXT_ARTIFACT_SCHEMA_VERSION}__"
            f"{STAGE6_ARTIFACT_SCHEMA_VERSION}__{STAGE6_RUNTIME_PROFILE_VERSION}__"
            f"de-{_decision_contract_tag()}__"
            f"seed-{seed}__progress.json"
        )
    )


def _new_progress(seed: int) -> dict[str, Any]:
    return {
        "case_set_version": MAIN_CASE_SET_VERSION,
        "snapshot_id": MAIN_SNAPSHOT_ID,
        "analysis_cutoff_date": MAIN_ANALYSIS_CUTOFF,
        "evaluation_mode": MAIN_EVALUATION_MODE,
        "context_scenario_set_version": CONTEXT_SCENARIO_SET_VERSION,
        "decision_evidence_retrieval_version": DECISION_EVIDENCE_RETRIEVAL_VERSION,
        "decision_source_registry_version": DECISION_SOURCE_REGISTRY_VERSION,
        "decision_source_snapshot_id": DECISION_SOURCE_SNAPSHOT_ID,
        "decision_source_manifest_sha256": DECISION_SOURCE_SPEC_MANIFEST_SHA256,
        "decision_source_records_sha256": DECISION_SOURCE_RECORDS_SHA256,
        "action_evidence_registry_version": ACTION_EVIDENCE_REGISTRY_VERSION,
        "action_evidence_snapshot_id": ACTION_EVIDENCE_SNAPSHOT_ID,
        "action_evidence_registry_sha256": ACTION_EVIDENCE_REGISTRY_SHA256,
        "main_guidance_selection_version": MAIN_GUIDANCE_SELECTION_VERSION,
        "main_guidance_selection_sha256": MAIN_GUIDANCE_SELECTION_SHA256,
        "stage4_context_artifact_schema_version": STAGE4_CONTEXT_ARTIFACT_SCHEMA_VERSION,
        "stage4_context_generation_contract_version": STAGE4_CONTEXT_GENERATION_CONTRACT_VERSION,
        "stage4_context_semantic_validation_version": STAGE4_CONTEXT_SEMANTIC_VALIDATION_VERSION,
        "stage4_context_runtime_profile_version": STAGE4_CONTEXT_RUNTIME_PROFILE_VERSION,
        "stage4_context_size": STAGE4_CONTEXT_SIZE,
        "stage4_server_context_size": STAGE4_SERVER_CONTEXT_SIZE,
        "stage4_parallel_slots": STAGE4_PARALLEL_SLOTS,
        "stage5_artifact_schema_version": STAGE5_ARTIFACT_SCHEMA_VERSION,
        "stage5_diagnostic_contract_version": STAGE5_DIAGNOSTIC_CONTRACT_VERSION,
        "stage6_artifact_schema_version": STAGE6_ARTIFACT_SCHEMA_VERSION,
        "stage6_synthesis_contract_version": STAGE6_SYNTHESIS_CONTRACT_VERSION,
        "stage6_decision_policy_version": STAGE6_DECISION_POLICY_VERSION,
        "stage6_semantic_validation_version": STAGE6_SEMANTIC_VALIDATION_VERSION,
        "stage6_generation_contract_version": STAGE6_GENERATION_CONTRACT_VERSION,
        "stage6_runtime_profile_version": STAGE6_RUNTIME_PROFILE_VERSION,
        "stage6_context_size": STAGE6_CONTEXT_SIZE,
        "stage6_server_context_size": STAGE6_SERVER_CONTEXT_SIZE,
        "stage6_parallel_slots": STAGE6_PARALLEL_SLOTS,
        "seed": seed,
        "last_updated_at": _now_iso(),
        "resume_contract": (
            "Exact Stage 1/2 artifacts, base Stage 3, contextual Stage 3 v4, and contextual Stage 4 v4 artifacts are "
            "reused first. Incomplete Stage 2, Stage 4 v4, or Stage 6 work resumes only from exact-fingerprint "
            "checkpoints. Both Stage 3 layers and Stage 5 are deterministic. Stage 6 uses constrained two-worker "
            "synthesis over frozen frames. Progress metadata never authorizes fallback."
        ),
        "cases": {},
    }


def _load_or_initialize_progress(path: Path, seed: int) -> dict[str, Any]:
    if not path.exists():
        return _new_progress(seed)
    try:
        progress = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return _new_progress(seed)
    expected = (
        MAIN_CASE_SET_VERSION,
        MAIN_SNAPSHOT_ID,
        MAIN_ANALYSIS_CUTOFF,
        MAIN_EVALUATION_MODE,
        CONTEXT_SCENARIO_SET_VERSION,
        DECISION_EVIDENCE_RETRIEVAL_VERSION,
        DECISION_SOURCE_REGISTRY_VERSION,
        DECISION_SOURCE_SNAPSHOT_ID,
        DECISION_SOURCE_SPEC_MANIFEST_SHA256,
        DECISION_SOURCE_RECORDS_SHA256,
        ACTION_EVIDENCE_REGISTRY_VERSION,
        ACTION_EVIDENCE_SNAPSHOT_ID,
        ACTION_EVIDENCE_REGISTRY_SHA256,
        MAIN_GUIDANCE_SELECTION_VERSION,
        MAIN_GUIDANCE_SELECTION_SHA256,
        STAGE4_CONTEXT_ARTIFACT_SCHEMA_VERSION,
        STAGE4_CONTEXT_GENERATION_CONTRACT_VERSION,
        STAGE4_CONTEXT_SEMANTIC_VALIDATION_VERSION,
        STAGE4_CONTEXT_RUNTIME_PROFILE_VERSION,
        STAGE4_CONTEXT_SIZE,
        STAGE4_SERVER_CONTEXT_SIZE,
        STAGE4_PARALLEL_SLOTS,
        STAGE5_ARTIFACT_SCHEMA_VERSION,
        STAGE5_DIAGNOSTIC_CONTRACT_VERSION,
        STAGE6_ARTIFACT_SCHEMA_VERSION,
        STAGE6_SYNTHESIS_CONTRACT_VERSION,
        STAGE6_DECISION_POLICY_VERSION,
        STAGE6_SEMANTIC_VALIDATION_VERSION,
        STAGE6_GENERATION_CONTRACT_VERSION,
        STAGE6_RUNTIME_PROFILE_VERSION,
        STAGE6_CONTEXT_SIZE,
        STAGE6_SERVER_CONTEXT_SIZE,
        STAGE6_PARALLEL_SLOTS,
        seed,
    )
    actual = (
        progress.get("case_set_version"),
        progress.get("snapshot_id"),
        progress.get("analysis_cutoff_date"),
        progress.get("evaluation_mode"),
        progress.get("context_scenario_set_version"),
        progress.get("decision_evidence_retrieval_version"),
        progress.get("decision_source_registry_version"),
        progress.get("decision_source_snapshot_id"),
        progress.get("decision_source_manifest_sha256"),
        progress.get("decision_source_records_sha256"),
        progress.get("action_evidence_registry_version"),
        progress.get("action_evidence_snapshot_id"),
        progress.get("action_evidence_registry_sha256"),
        progress.get("main_guidance_selection_version"),
        progress.get("main_guidance_selection_sha256"),
        progress.get("stage4_context_artifact_schema_version"),
        progress.get("stage4_context_generation_contract_version"),
        progress.get("stage4_context_semantic_validation_version"),
        progress.get("stage4_context_runtime_profile_version"),
        progress.get("stage4_context_size"),
        progress.get("stage4_server_context_size"),
        progress.get("stage4_parallel_slots"),
        progress.get("stage5_artifact_schema_version"),
        progress.get("stage5_diagnostic_contract_version"),
        progress.get("stage6_artifact_schema_version"),
        progress.get("stage6_synthesis_contract_version"),
        progress.get("stage6_decision_policy_version"),
        progress.get("stage6_semantic_validation_version"),
        progress.get("stage6_generation_contract_version"),
        progress.get("stage6_runtime_profile_version"),
        progress.get("stage6_context_size"),
        progress.get("stage6_server_context_size"),
        progress.get("stage6_parallel_slots"),
        progress.get("seed"),
    )
    if actual != expected:
        return _new_progress(seed)
    progress.setdefault("cases", {})
    return progress


def _record_progress(path: Path, progress: dict[str, Any]) -> None:
    progress["last_updated_at"] = _now_iso()
    _write_json_atomic(path, progress)


def _print_runtime_profile() -> None:
    runtime = get_runtime_config()
    print(
        "Shared model/sampling profile: "
        f"{runtime.model} via llama.cpp at {runtime.api_base} "
        f"(thinking={runtime.reasoning_effort}, temp={runtime.temperature}, "
        f"top_p={runtime.top_p}, top_k={runtime.top_k}, seed={runtime.seed})",
        flush=True,
    )
    print(
        f"Active shared runtime: {STAGE4_SERVER_CONTEXT_SIZE} total tokens / "
        f"{runtime.expected_parallel_slots} slots = {runtime.expected_context_size} tokens per slot; "
        f"contextual Stage 4 identity={STAGE4_CONTEXT_RUNTIME_PROFILE_VERSION}; "
        f"Stage 6 synthesis identity={STAGE6_RUNTIME_PROFILE_VERSION}. "
        "Historical one-slot Stage 1/2 artifacts may be reused through the explicit concurrency-only compatibility path.",
        flush=True,
    )


async def _run_main_case(
    *,
    app: Any,
    case: MainStage1Case,
    index: int,
    total: int,
    progress: dict[str, Any],
    progress_path: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    print("\n" + "=" * 100, flush=True)
    print(f"[MAIN SET][{index}/{total}] {case.case_id}", flush=True)
    print(f"Threat={case.threat} | PMT={case.pmt}", flush=True)
    print(f"Selection={case.selection_class} | Regime={case.regime or 'anchor'}", flush=True)
    print(f"Reason={case.selection_reason}", flush=True)
    print("=" * 100, flush=True)

    progress["cases"][case.case_id] = {
        "index": index,
        "threat": case.threat,
        "pmt": case.pmt,
        "status": "running",
        "started_at": _now_iso(),
        "selection_class": case.selection_class,
        "regime": case.regime,
    }
    _record_progress(progress_path, progress)

    try:
        result = await app.ainvoke(
            _initial_state(_case_config(case)),
            config={"recursion_limit": 100},
        )
    except Exception as exc:
        progress["cases"][case.case_id].update(
            {
                "status": "failed",
                "failed_at": _now_iso(),
                "error": repr(exc),
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
        )
        _record_progress(progress_path, progress)
        print(f"[MAIN SET][{index}/{total}] FAILED: {exc}", flush=True)
        raise

    pack = result.get("evidence_pack") or {}
    retrieval = pack.get("retrieval_metadata") or {}
    reused_stage1 = bool(result.get("stage1_artifact_reused"))
    reused_stage2 = bool(result.get("stage2_artifact_reused"))
    reused_stage3 = bool(result.get("stage3_artifact_reused"))
    reused_stage3_context = bool(result.get("stage3_context_artifact_reused"))
    reused_stage4_context = bool(result.get("stage4_context_artifact_reused"))
    reused_stage5 = bool(result.get("stage5_artifact_reused"))
    reused_stage6 = bool(result.get("stage6_artifact_reused"))
    action = (
        "reused_stage6_synthesis_artifact"
        if reused_stage6
        else "generated_or_resumed_stage6_and_frozen"
    )
    elapsed = time.monotonic() - started
    progress["cases"][case.case_id].update(
        {
            "status": "complete",
            "completed_at": _now_iso(),
            "last_action": action,
            "stage1_artifact_reused": reused_stage1,
            "stage2_artifact_reused": reused_stage2,
            "attack_checkpoint_reused": bool(result.get("attack_checkpoint_reused")),
            "defense_checkpoint_reused": bool(result.get("defense_checkpoint_reused")),
            "stage1_input_fingerprint": result.get("stage1_input_fingerprint"),
            "stage1_artifact_path": result.get("stage1_artifact_path"),
            "stage2_input_fingerprint": result.get("stage2_input_fingerprint"),
            "stage2_artifact_path": result.get("stage2_artifact_path"),
            "stage3_artifact_reused": reused_stage3,
            "stage3_input_fingerprint": result.get("stage3_input_fingerprint"),
            "stage3_artifact_path": result.get("stage3_artifact_path"),
            "decision_object_count": len(result.get("decision_objects") or []),
            "stage3_context_artifact_reused": reused_stage3_context,
            "stage3_context_input_fingerprint": result.get("stage3_context_input_fingerprint"),
            "stage3_context_artifact_path": result.get("stage3_context_artifact_path"),
            "contextual_decision_object_count": len(result.get("contextual_decision_objects") or []),
            "stage4_context_artifact_reused": reused_stage4_context,
            "stage4_context_input_fingerprint": result.get("stage4_context_input_fingerprint"),
            "stage4_context_artifact_path": result.get("stage4_context_artifact_path"),
            "stage4_context_evaluation_count": len(
                (result.get("stage4_context_evaluation_bundle") or {}).get("evaluations") or []
            ),
            "stage5_artifact_reused": reused_stage5,
            "stage5_input_fingerprint": result.get("stage5_input_fingerprint"),
            "stage5_artifact_path": result.get("stage5_artifact_path"),
            "stage5_scenario_diagnostic_count": len(
                (result.get("stage5_diagnostic_bundle") or {}).get("scenario_diagnostics") or []
            ),
            "stage6_artifact_reused": reused_stage6,
            "stage6_input_fingerprint": result.get("stage6_input_fingerprint"),
            "stage6_artifact_path": result.get("stage6_artifact_path"),
            "stage6_synthesis_count": len(
                (result.get("stage6_synthesis_bundle") or {}).get("syntheses") or []
            ),
            "stage6_checkpoint_reuse_count": result.get("stage6_checkpoint_reuse_count", 0),
            "contextual_technical_checkpoint_reuse_count": result.get(
                "contextual_technical_checkpoint_reuse_count", 0
            ),
            "contextual_institutional_checkpoint_reuse_count": result.get(
                "contextual_institutional_checkpoint_reuse_count", 0
            ),
            "contextual_financial_checkpoint_reuse_count": result.get(
                "contextual_financial_checkpoint_reuse_count", 0
            ),
            "debate_round_count": len(result.get("debate_rounds") or []),
            "retrieved_evidence_count": len(pack.get("evidence") or []),
            "covered_evidence_slots": retrieval.get("covered_evidence_slots"),
            "missing_evidence_slots": retrieval.get("missing_evidence_slots"),
            "source_family_count": retrieval.get("source_family_count"),
            "evidence_sufficiency": retrieval.get("evidence_sufficiency"),
            "elapsed_seconds": round(elapsed, 3),
        }
    )
    _record_progress(progress_path, progress)
    print(
        f"[MAIN SET][{index}/{total}] COMPLETE in {elapsed:.1f}s | "
        f"action={action} | evidence={len(pack.get('evidence') or [])} | "
        f"slots={len(retrieval.get('covered_evidence_slots') or [])}/6 | "
        f"families={retrieval.get('source_family_count')} | "
        f"rounds={len(result.get('debate_rounds') or [])} | "
        f"stage2_artifact={result.get('stage2_artifact_path')} | "
        f"stage3_objects={len(result.get('decision_objects') or [])} | "
        f"stage3_artifact={result.get('stage3_artifact_path')} | "
        f"stage3_context_objects={len(result.get('contextual_decision_objects') or [])} | "
        f"stage3_context_artifact={result.get('stage3_context_artifact_path')} | "
        f"stage4_context_evaluations={len((result.get('stage4_context_evaluation_bundle') or {}).get('evaluations') or [])} | "
        f"stage4_context_artifact={result.get('stage4_context_artifact_path')} | "
        f"stage5_artifact={result.get('stage5_artifact_path')} | "
        f"stage6_syntheses={len((result.get('stage6_synthesis_bundle') or {}).get('syntheses') or [])} | "
        f"stage6_artifact={result.get('stage6_artifact_path')}",
        flush=True,
    )
    return result


async def _run_main_set() -> None:
    validate_main_case_set()
    root = _project_root()
    case_manifest_path = _ensure_frozen_case_set_manifest(root)
    preflight_path = _preflight_main_decision_evidence(root)
    runtime = get_runtime_config()
    progress_path = _progress_path(root, runtime.seed)
    progress = _load_or_initialize_progress(progress_path, runtime.seed)

    print(
        "--- Stage 0 -> Stage 1 -> Stage 2 -> Stage 3 base -> Stage 3 context v4 -> Stage 4 v4 -> Stage 5 diagnostics -> Stage 6 synthesis: batch execution ---",
        flush=True,
    )
    _print_runtime_profile()
    print(f"Case-set manifest: {case_manifest_path}", flush=True)
    print(f"Decision-evidence preflight: {preflight_path}", flush=True)
    print(f"Progress/resume state: {progress_path}", flush=True)
    print(
        "Resume is automatic: exact Stage 1/2/base-Stage3/contextual-Stage3/Stage4-v4/Stage5/Stage6 artifacts are "
        "reused; interrupted Stage 1 critics, Stage 2 debate steps, contextual Stage 4 lenses, and Stage 6 syntheses "
        "resume only from exact-fingerprint checkpoints. Both Stage 3 layers and Stage 5 remain deterministic.",
        flush=True,
    )

    app = create_graph()
    batch_started = time.monotonic()
    for index, case in enumerate(MAIN_STAGE1_CASES, start=1):
        await _run_main_case(
            app=app,
            case=case,
            index=index,
            total=len(MAIN_STAGE1_CASES),
            progress=progress,
            progress_path=progress_path,
        )

    elapsed = time.monotonic() - batch_started
    statuses = list(progress["cases"].values())
    reused = sum(item.get("last_action") == "reused_stage6_synthesis_artifact" for item in statuses)
    generated = sum(
        item.get("last_action") == "generated_or_resumed_stage6_and_frozen"
        for item in statuses
    )
    print("\n" + "=" * 100, flush=True)
    print(
        f"MAIN SET COMPLETE: {len(MAIN_STAGE1_CASES)}/{len(MAIN_STAGE1_CASES)} cases | "
        f"generated/resumed={generated} | reused={reused} | total={elapsed / 60:.1f} min",
        flush=True,
    )
    print(f"Progress file: {progress_path}", flush=True)
    print("=" * 100, flush=True)


async def _run_single_case() -> None:
    print("--- Single Stage 0 -> Stage 1 -> Stage 2 -> Stage 3 v4 -> Stage 4 v4 -> Stage 5 -> Stage 6 case execution ---", flush=True)
    _print_runtime_profile()
    app = create_graph()
    started = time.monotonic()
    result = await app.ainvoke(_initial_state(), config={"recursion_limit": 100})
    elapsed = time.monotonic() - started
    print(
        f"Single case complete in {elapsed:.1f}s | stage1={result.get('stage1_artifact_path')} | "
        f"stage2={result.get('stage2_artifact_path')} | stage3={result.get('stage3_artifact_path')} | "
        f"stage3_context={result.get('stage3_context_artifact_path')} | "
        f"stage4_context={result.get('stage4_context_artifact_path')} | "
        f"stage5={result.get('stage5_artifact_path')} | stage6={result.get('stage6_artifact_path')}",
        flush=True,
    )


async def _run_smoke_critics(*, count: int, sample_seed: int) -> None:
    """Run a reproducible random subset of individual critics without freezing main artifacts."""
    validate_main_case_set()
    population: list[tuple[MainStage1Case, CriticType, str]] = []
    for case in MAIN_STAGE1_CASES:
        population.extend(
            [
                (case, CriticType.ATTACK_FEASIBILITY, ATTACK_FEASIBILITY_SYSTEM_PROMPT),
                (case, CriticType.DEFENSE_ROBUSTNESS, DEFENSE_ROBUSTNESS_SYSTEM_PROMPT),
            ]
        )
    if not 1 <= count <= len(population):
        raise ValueError(f"--smoke-critics must be between 1 and {len(population)}")

    selected = random.Random(sample_seed).sample(population, count)
    print("--- Stage 1 random critic smoke evaluation ---", flush=True)
    _print_runtime_profile()
    print(
        f"Sampling {count}/{len(population)} critics without replacement using selection seed={sample_seed}. "
        "Smoke runs do not create or reuse main Stage 1 artifacts/checkpoints.",
        flush=True,
    )

    outputs: list[dict[str, Any]] = []
    started = time.monotonic()
    for index, (case, critic_type, prompt_template) in enumerate(selected, start=1):
        print(
            f"\n[SMOKE][{index}/{count}] {case.case_id} | critic={critic_type.value} | "
            f"Threat={case.threat} | PMT={case.pmt}",
            flush=True,
        )
        agent_input = build_agent_input(
            project_root=_project_root(),
            snapshot_id=MAIN_SNAPSHOT_ID,
            threat=case.threat,
            pmt=case.pmt,
            analysis_cutoff_date=MAIN_ANALYSIS_CUTOFF,
            evaluation_mode=MAIN_EVALUATION_MODE,
            case_id=case.case_id,
        )
        assessment = await _run_stage1_critic(
            {
                "forecast_data": agent_input["forecast_data"],
                "evidence_pack": agent_input["evidence_pack"],
            },
            critic_type=critic_type,
            prompt_template=prompt_template,
        )
        relation_counts: dict[str, int] = {}
        for claim in assessment["claims"]:
            relation = claim["forecast_relation"]
            relation_counts[relation] = relation_counts.get(relation, 0) + 1
        print(
            f"[SMOKE][{index}/{count}] stance={assessment['stance']} | "
            f"decision={assessment['stance_basis']['decision']} | claims={len(assessment['claims'])} | "
            f"relations={relation_counts} | sufficiency={assessment['evidence_sufficiency']}",
            flush=True,
        )
        outputs.append(
            {
                "case_id": case.case_id,
                "threat": case.threat,
                "pmt": case.pmt,
                "critic_type": critic_type.value,
                "assessment": assessment,
            }
        )

    root = _project_root()
    smoke_dir = root / "Multi-Agent" / "Results" / "Stage1" / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = smoke_dir / (
        f"{STAGE1_ARTIFACT_SCHEMA_VERSION}__sample-{count}__selection-seed-{sample_seed}__{timestamp}.json"
    )
    _write_json_atomic(
        output_path,
        {
            "schema_version": STAGE1_ARTIFACT_SCHEMA_VERSION,
            "sample_selection_seed": sample_seed,
            "sample_size": count,
            "created_at": _now_iso(),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "outputs": outputs,
        },
    )
    print(f"\nSMOKE COMPLETE: {count} critic(s) | output={output_path}", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen seven-case Stage 0 -> Stage 1 -> Stage 2 -> Stage 3 contextual v4 -> Stage 4 v4 -> Stage 5 -> Stage 6 "
            "experiment set, or one env-configured case."
        )
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Run one case using STAGE0_* environment variables instead of the frozen seven-case main set.",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="Print the frozen main case set and exit without inference.",
    )
    parser.add_argument(
        "--smoke-critics",
        type=int,
        default=0,
        metavar="N",
        help="Randomly sample N of the 14 case/critic combinations and run only those critics without main artifacts.",
    )
    parser.add_argument(
        "--smoke-seed",
        type=int,
        default=20260913,
        help="Selection seed for --smoke-critics. This does not change the frozen LLM inference seed.",
    )
    return parser.parse_args()


async def main(args: argparse.Namespace) -> None:
    if args.list_cases:
        print(json.dumps(main_case_set_manifest(), ensure_ascii=False, indent=2))
        return
    if args.single and args.smoke_critics:
        raise ValueError("--single and --smoke-critics are mutually exclusive")

    root = _project_root()
    log_stage = "Stage1" if args.smoke_critics else "Stage6"
    log_dir = root / "Multi-Agent" / "Results" / log_stage / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = (
        "single"
        if args.single
        else f"smoke-{args.smoke_critics}-selection-seed-{args.smoke_seed}"
        if args.smoke_critics
        else MAIN_CASE_SET_VERSION
    )
    log_path = log_dir / f"{run_name}__{timestamp}.log"

    with log_path.open("a", encoding="utf-8", buffering=1) as log_file:
        tee = _TeeTextIO(sys.stdout, log_file)
        with redirect_stdout(tee), redirect_stderr(tee):
            print(f"Run log: {log_path}", flush=True)
            try:
                if args.single:
                    await _run_single_case()
                elif args.smoke_critics:
                    await _run_smoke_critics(count=args.smoke_critics, sample_seed=args.smoke_seed)
                else:
                    await _run_main_set()
            except Exception:
                print(
                    "Run interrupted/failed. Re-run the same command after fixing the cause; exact Stage 1/2/3/4/5/6 "
                    "artifacts and exact Stage 1/2/4/6 checkpoints resume without fallback or overwrite.",
                    flush=True,
                )
                raise


if __name__ == "__main__":
    asyncio.run(main(_parse_args()))
