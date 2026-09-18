from __future__ import annotations

import re
from pathlib import Path
from typing import Any


_COMPLETION_RE = re.compile(
    r"\[QWEN\]\[(?P<label>[^\]]+)\] completed\s+"
    r"elapsed=(?P<elapsed>[0-9.]+)s"
    r"(?:\s+prompt_tokens=(?P<prompt>\d+))?"
    r"(?:\s+completion_tokens=(?P<completion>\d+))?"
    r"(?:\s+total_tokens=(?P<total>\d+))?"
    r"(?:\s+reasoning_tokens=(?P<reasoning>\d+))?"
    r"(?:\s+reasoning_chars=(?P<reasoning_chars>\d+))?"
)
_REPAIR_RE = re.compile(
    r"\[QWEN\]\[(?P<label>[^\]]+)\] structured schema/semantic validation failed "
    r"\(attempt (?P<attempt>\d+)/(?:\d+)\): (?P<error_type>[^:]+): (?P<message>.*)"
)
_MAIN_COMPLETE_RE = re.compile(r"MAIN SET COMPLETE:\s+7/7 cases")
_MAIN_FAIL_RE = re.compile(r"\[MAIN SET\]\[(?P<index>\d+)/7\] FAILED:\s+(?P<message>.*)")


def _stage_from_label(label: str) -> str:
    upper = label.upper()
    for stage in ("STAGE7", "STAGE6", "STAGE4", "STAGE2", "STAGE1"):
        if stage in upper:
            return stage.title().replace("Stage", "Stage")
    return "UNKNOWN"


def parse_run_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    completions: list[dict[str, Any]] = []
    for match in _COMPLETION_RE.finditer(text):
        item = match.groupdict()
        completions.append(
            {
                "label": item["label"],
                "stage": _stage_from_label(item["label"]),
                "elapsed_seconds": float(item["elapsed"]),
                "prompt_tokens": int(item["prompt"]) if item.get("prompt") else None,
                "completion_tokens": int(item["completion"]) if item.get("completion") else None,
                "total_tokens": int(item["total"]) if item.get("total") else None,
                "reasoning_tokens": int(item["reasoning"]) if item.get("reasoning") else None,
                "reasoning_chars": int(item["reasoning_chars"]) if item.get("reasoning_chars") else None,
                "is_repair": ":repair-" in item["label"],
            }
        )
    repairs = [
        {
            "label": match.group("label"),
            "stage": _stage_from_label(match.group("label")),
            "attempt": int(match.group("attempt")),
            "error_type": match.group("error_type"),
            "message": match.group("message"),
        }
        for match in _REPAIR_RE.finditer(text)
    ]
    failures = [match.groupdict() for match in _MAIN_FAIL_RE.finditer(text)]
    return {
        "path": str(path),
        "main_set_complete": bool(_MAIN_COMPLETE_RE.search(text)),
        "completion_events": completions,
        "repair_events": repairs,
        "hard_fail_events": failures,
    }


def stage6_run_logs(root: Path) -> list[Path]:
    folder = root / "Multi-Agent" / "Results" / "Stage6" / "logs"
    if not folder.exists():
        return []
    return sorted(folder.glob("*.log"), key=lambda p: p.stat().st_mtime)


def latest_successful_main_log(root: Path) -> dict[str, Any] | None:
    for path in reversed(stage6_run_logs(root)):
        parsed = parse_run_log(path)
        if parsed["main_set_complete"]:
            return parsed
    return None


def aggregate_completion_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    def total(key: str) -> int:
        return sum(int(item[key]) for item in events if item.get(key) is not None)

    by_stage: dict[str, dict[str, Any]] = {}
    for item in events:
        stage = item["stage"]
        payload = by_stage.setdefault(
            stage,
            {
                "calls": 0,
                "repair_calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "elapsed_seconds_sum": 0.0,
            },
        )
        payload["calls"] += 1
        payload["repair_calls"] += int(item["is_repair"])
        payload["elapsed_seconds_sum"] += float(item["elapsed_seconds"])
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            if item.get(key) is not None:
                payload[key] += int(item[key])
    return {
        "calls": len(events),
        "repair_calls": sum(int(item["is_repair"]) for item in events),
        "prompt_tokens": total("prompt_tokens"),
        "completion_tokens": total("completion_tokens"),
        "total_tokens": total("total_tokens"),
        "elapsed_seconds_sum": sum(float(item["elapsed_seconds"]) for item in events),
        "by_stage": by_stage,
    }
