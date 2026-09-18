from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from Stage1.cases import MAIN_STAGE1_CASES
from Stage6.schema import STAGE6_SEMANTIC_VALIDATION_VERSION


class Stage7InputError(ValueError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Stage7InputError(f"Expected JSON object at {path}")
    return value


def verify_artifact_hash(value: dict[str, Any], *, path: Path | None = None) -> str:
    stored = str(value.get("artifact_sha256") or "")
    if not stored:
        raise Stage7InputError(f"Artifact missing artifact_sha256: {path or '<memory>'}")
    unhashed = {key: item for key, item in value.items() if key != "artifact_sha256"}
    actual = sha256_json(unhashed)
    if stored != actual:
        raise Stage7InputError(
            f"Artifact hash mismatch at {path or '<memory>'}: stored={stored} actual={actual}"
        )
    return stored


def _artifact_files(root: Path, relative: str, case_id: str) -> list[Path]:
    folder = root / "Multi-Agent" / "Results" / relative / case_id
    if not folder.exists():
        return []
    return sorted(folder.glob("*.json"))


def _find_artifact_by_hash(
    root: Path,
    *,
    relative: str,
    case_id: str,
    artifact_sha256: str,
) -> tuple[dict[str, Any], Path]:
    for path in _artifact_files(root, relative, case_id):
        value = load_json(path)
        if str(value.get("artifact_sha256") or "") != artifact_sha256:
            continue
        verify_artifact_hash(value, path=path)
        return value, path
    raise Stage7InputError(
        f"Could not resolve exact {relative} artifact for {case_id}: {artifact_sha256}"
    )


def _created_at_key(value: dict[str, Any]) -> str:
    return str(value.get("created_at") or "")


def discover_final_stage6_artifact(root: Path, case_id: str) -> tuple[dict[str, Any], Path]:
    candidates: list[tuple[dict[str, Any], Path]] = []
    for path in _artifact_files(root, "Stage6/artifacts", case_id):
        value = load_json(path)
        if value.get("semantic_validation_version") != STAGE6_SEMANTIC_VALIDATION_VERSION:
            continue
        verify_artifact_hash(value, path=path)
        candidates.append((value, path))
    if not candidates:
        raise Stage7InputError(
            f"No current Stage 6 artifact ({STAGE6_SEMANTIC_VALIDATION_VERSION}) for {case_id}"
        )
    candidates.sort(key=lambda item: _created_at_key(item[0]))
    return candidates[-1]


@dataclass(frozen=True)
class CaseBundle:
    case_id: str
    stage1: dict[str, Any]
    stage1_path: Path
    stage2: dict[str, Any]
    stage2_path: Path
    stage3_context: dict[str, Any]
    stage3_context_path: Path
    stage4_context: dict[str, Any]
    stage4_context_path: Path
    stage5: dict[str, Any]
    stage5_path: Path
    stage6: dict[str, Any]
    stage6_path: Path

    @property
    def syntheses(self) -> list[dict[str, Any]]:
        return list((self.stage6.get("synthesis_bundle") or {}).get("syntheses") or [])

    @property
    def stage2_result(self) -> dict[str, Any]:
        return dict(self.stage2.get("debate_result") or {})

    @property
    def contextual_objects(self) -> list[dict[str, Any]]:
        return list(
            (self.stage3_context.get("contextual_decision_bundle") or {}).get(
                "contextual_decision_objects"
            )
            or []
        )

    @property
    def stage4_evaluations(self) -> list[dict[str, Any]]:
        return list((self.stage4_context.get("evaluation_bundle") or {}).get("evaluations") or [])

    def evidence_records_by_id(self) -> dict[str, dict[str, Any]]:
        records: dict[str, dict[str, Any]] = {}
        for obj in self.contextual_objects:
            for record in obj.get("evaluation_evidence_records") or []:
                evidence_id = str(record.get("evidence_id") or "")
                if not evidence_id:
                    continue
                existing = records.get(evidence_id)
                if existing is not None and canonical_json(existing) != canonical_json(record):
                    raise Stage7InputError(
                        f"Evidence record drift across contextual scenarios for {self.case_id}: {evidence_id}"
                    )
                records[evidence_id] = record
        return records

    def synthesis_by_scenario(self) -> dict[str, dict[str, Any]]:
        result = {str(item["scenario_id"]): item for item in self.syntheses}
        if len(result) != len(self.syntheses):
            raise Stage7InputError(f"Duplicate Stage 6 scenario_id in {self.case_id}")
        return result

    def contextual_object_by_scenario(self) -> dict[str, dict[str, Any]]:
        result = {str(item["scenario_id"]): item for item in self.contextual_objects}
        if len(result) != len(self.contextual_objects):
            raise Stage7InputError(f"Duplicate contextual scenario_id in {self.case_id}")
        return result

    def stage4_by_decision_object(self) -> dict[str, dict[str, Any]]:
        return {str(item["decision_object_id"]): item for item in self.stage4_evaluations}

    def source_hashes(self) -> dict[str, str]:
        return {
            "stage1": str(self.stage1.get("artifact_sha256") or ""),
            "stage2": str(self.stage2.get("artifact_sha256") or ""),
            "stage3_context": str(self.stage3_context.get("artifact_sha256") or ""),
            "stage4_context": str(self.stage4_context.get("artifact_sha256") or ""),
            "stage5": str(self.stage5.get("artifact_sha256") or ""),
            "stage6": str(self.stage6.get("artifact_sha256") or ""),
        }


def load_case_bundle(root: Path, case_id: str) -> CaseBundle:
    stage6, stage6_path = discover_final_stage6_artifact(root, case_id)
    stage3, stage3_path = _find_artifact_by_hash(
        root,
        relative="Stage3/contextual_artifacts",
        case_id=case_id,
        artifact_sha256=str(stage6["stage3_context_artifact_sha256"]),
    )
    stage4, stage4_path = _find_artifact_by_hash(
        root,
        relative="Stage4/contextual_artifacts",
        case_id=case_id,
        artifact_sha256=str(stage6["stage4_context_artifact_sha256"]),
    )
    stage5, stage5_path = _find_artifact_by_hash(
        root,
        relative="Stage5/artifacts",
        case_id=case_id,
        artifact_sha256=str(stage6["stage5_artifact_sha256"]),
    )
    # Cross-check Stage 6 -> Stage 5 exact provenance, not only directory names.
    if stage5.get("stage3_context_artifact_sha256") != stage3.get("artifact_sha256"):
        raise Stage7InputError(f"Stage 5 -> Stage 3 hash mismatch for {case_id}")
    if stage5.get("stage4_context_artifact_sha256") != stage4.get("artifact_sha256"):
        raise Stage7InputError(f"Stage 5 -> Stage 4 hash mismatch for {case_id}")

    stage2, stage2_path = _find_artifact_by_hash(
        root,
        relative="Stage2/artifacts",
        case_id=case_id,
        artifact_sha256=str(stage5["stage2_artifact_sha256"]),
    )
    stage1, stage1_path = _find_artifact_by_hash(
        root,
        relative="Stage1/artifacts",
        case_id=case_id,
        artifact_sha256=str(stage2["stage1_artifact_sha256"]),
    )

    if stage1.get("case_id") != case_id or stage2.get("case_id") != case_id:
        raise Stage7InputError(f"Case ID drift in Stage 1/2 artifacts for {case_id}")
    if stage3.get("case_id") != case_id or stage4.get("case_id") != case_id:
        raise Stage7InputError(f"Case ID drift in Stage 3/4 artifacts for {case_id}")
    if stage5.get("case_id") != case_id or stage6.get("case_id") != case_id:
        raise Stage7InputError(f"Case ID drift in Stage 5/6 artifacts for {case_id}")

    bundle = CaseBundle(
        case_id=case_id,
        stage1=stage1,
        stage1_path=stage1_path,
        stage2=stage2,
        stage2_path=stage2_path,
        stage3_context=stage3,
        stage3_context_path=stage3_path,
        stage4_context=stage4,
        stage4_context_path=stage4_path,
        stage5=stage5,
        stage5_path=stage5_path,
        stage6=stage6,
        stage6_path=stage6_path,
    )
    # Force evidence consistency validation during load.
    bundle.evidence_records_by_id()
    if len(bundle.syntheses) != 3:
        raise Stage7InputError(f"Expected exactly 3 Stage 6 scenarios for {case_id}")
    return bundle


def load_main_case_bundles(root: Path) -> list[CaseBundle]:
    bundles = [load_case_bundle(root, case.case_id) for case in MAIN_STAGE1_CASES]
    if len(bundles) != len(MAIN_STAGE1_CASES):
        raise Stage7InputError("Main case bundle count mismatch")
    return bundles


def iter_stage7_input_files(root: Path) -> Iterable[Path]:
    folder = root / "Multi-Agent" / "Stage7" / "Inputs"
    if not folder.exists():
        return []
    return sorted(path for path in folder.glob("*.json") if path.is_file())
