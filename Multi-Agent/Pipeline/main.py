from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import os
import sys
import threading
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, TextIO

from dotenv import load_dotenv

from Pipeline.graph import create_graph
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
from Stage1.runtime import get_runtime_config


load_dotenv()


class _TeeTextIO:
    """Line-buffered console + file sink so interrupted runs keep their logs."""

    def __init__(self, terminal: TextIO, log_file: TextIO):
        self.terminal = terminal
        self.log_file = log_file
        self._lock = threading.Lock()

    def write(self, text: str) -> int:
        with self._lock:
            self.terminal.write(text)
            self.terminal.flush()
            self.log_file.write(text)
            self.log_file.flush()
        return len(text)

    def flush(self) -> None:
        with self._lock:
            self.terminal.flush()
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
    path = root / "Data" / "Stage1" / "case_sets" / f"{MAIN_CASE_SET_VERSION}.json"
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
        "messages": [],
        "iteration_count": 0,
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


def _progress_path(root: Path, seed: int) -> Path:
    return (
        root
        / "Data"
        / "Stage1"
        / "runs"
        / f"{MAIN_CASE_SET_VERSION}__seed-{seed}__progress.json"
    )


def _new_progress(seed: int) -> dict[str, Any]:
    return {
        "case_set_version": MAIN_CASE_SET_VERSION,
        "snapshot_id": MAIN_SNAPSHOT_ID,
        "analysis_cutoff_date": MAIN_ANALYSIS_CUTOFF,
        "evaluation_mode": MAIN_EVALUATION_MODE,
        "seed": seed,
        "last_updated_at": _now_iso(),
        "resume_contract": (
            "Exact final artifacts are reused first. If a final artifact is absent, exact-fingerprint "
            "per-critic checkpoints are reused independently. Progress metadata never authorizes fallback."
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
    expected = (MAIN_CASE_SET_VERSION, MAIN_SNAPSHOT_ID, MAIN_ANALYSIS_CUTOFF, MAIN_EVALUATION_MODE, seed)
    actual = (
        progress.get("case_set_version"),
        progress.get("snapshot_id"),
        progress.get("analysis_cutoff_date"),
        progress.get("evaluation_mode"),
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
        "Stage 1 runtime profile: "
        f"{runtime.model} via llama.cpp at {runtime.api_base} "
        f"(thinking={runtime.reasoning_effort}, temp={runtime.temperature}, "
        f"top_p={runtime.top_p}, top_k={runtime.top_k}, seed={runtime.seed})",
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
    reused_final = bool(result.get("stage1_artifact_reused"))
    action = "reused_final_artifact" if reused_final else "generated_or_resumed_and_frozen"
    elapsed = time.monotonic() - started
    progress["cases"][case.case_id].update(
        {
            "status": "complete",
            "completed_at": _now_iso(),
            "last_action": action,
            "final_artifact_reused": reused_final,
            "attack_checkpoint_reused": bool(result.get("attack_checkpoint_reused")),
            "defense_checkpoint_reused": bool(result.get("defense_checkpoint_reused")),
            "input_fingerprint": result.get("stage1_input_fingerprint"),
            "artifact_path": result.get("stage1_artifact_path"),
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
        f"artifact={result.get('stage1_artifact_path')}",
        flush=True,
    )
    return result


async def _run_main_set() -> None:
    validate_main_case_set()
    root = _project_root()
    case_manifest_path = _ensure_frozen_case_set_manifest(root)
    runtime = get_runtime_config()
    progress_path = _progress_path(root, runtime.seed)
    progress = _load_or_initialize_progress(progress_path, runtime.seed)

    print("--- Stage 1 Main Case Set: batch execution ---", flush=True)
    _print_runtime_profile()
    print(f"Case-set manifest: {case_manifest_path}", flush=True)
    print(f"Progress/resume state: {progress_path}", flush=True)
    print(
        "Resume is automatic: exact final artifacts are reused; if a case stopped after one critic, "
        "that critic's exact checkpoint is reused and only the missing critic is generated.",
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
    reused = sum(item.get("last_action") == "reused_final_artifact" for item in statuses)
    generated = sum(item.get("last_action") == "generated_or_resumed_and_frozen" for item in statuses)
    print("\n" + "=" * 100, flush=True)
    print(
        f"MAIN SET COMPLETE: {len(MAIN_STAGE1_CASES)}/{len(MAIN_STAGE1_CASES)} cases | "
        f"generated/resumed={generated} | reused={reused} | total={elapsed / 60:.1f} min",
        flush=True,
    )
    print(f"Progress file: {progress_path}", flush=True)
    print("=" * 100, flush=True)


async def _run_single_case() -> None:
    print("--- Single Stage 0 -> Stage 1 case execution ---", flush=True)
    _print_runtime_profile()
    app = create_graph()
    started = time.monotonic()
    result = await app.ainvoke(_initial_state(), config={"recursion_limit": 100})
    elapsed = time.monotonic() - started
    print(
        f"Single case complete in {elapsed:.1f}s | artifact={result.get('stage1_artifact_path')}",
        flush=True,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen seven-case Stage 1 main experiment set, or one env-configured case."
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
    return parser.parse_args()


async def main(args: argparse.Namespace) -> None:
    if args.list_cases:
        print(json.dumps(main_case_set_manifest(), ensure_ascii=False, indent=2))
        return

    root = _project_root()
    log_dir = root / "Data" / "Stage1" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = "single" if args.single else MAIN_CASE_SET_VERSION
    log_path = log_dir / f"{run_name}__{timestamp}.log"

    with log_path.open("a", encoding="utf-8", buffering=1) as log_file:
        tee = _TeeTextIO(sys.stdout, log_file)
        with redirect_stdout(tee), redirect_stderr(tee):
            print(f"Run log: {log_path}", flush=True)
            try:
                if args.single:
                    await _run_single_case()
                else:
                    await _run_main_set()
            except Exception:
                print(
                    "Run interrupted/failed. Re-run the same command after fixing the cause; exact artifacts and "
                    "per-critic checkpoints will resume without fallback or overwrite.",
                    flush=True,
                )
                raise


if __name__ == "__main__":
    asyncio.run(main(_parse_args()))
