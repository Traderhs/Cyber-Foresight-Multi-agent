from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from Stage5.schema import (
    STAGE5_DIAGNOSTIC_CONTRACT_VERSION,
    Stage5DiagnosticBundle,
    Stage5ValidationError,
)


STAGE5_ARTIFACT_SCHEMA_VERSION = "stage5-artifact-v1"


class Stage5ArtifactError(ValueError):
    """Raised when a frozen Stage 5 artifact violates exact provenance."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_text(_canonical_json(value))


def _safe(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return safe or "item"


def _require_valid_artifact_hash(artifact: dict[str, Any], *, label: str) -> str:
    stored = str(artifact.get("artifact_sha256") or "")
    if not stored:
        raise Stage5ArtifactError(f"Stage 5 requires {label} artifact_sha256")
    unhashed = {key: value for key, value in artifact.items() if key != "artifact_sha256"}
    if stored != _sha256_json(unhashed):
        raise Stage5ArtifactError(f"Stage 5 rejected {label} artifact content hash mismatch")
    return stored


def build_stage5_identity(
    *,
    evidence_pack: dict[str, Any],
    stage2_artifact: dict[str, Any],
    stage3_context_artifact: dict[str, Any],
    stage4_context_artifact: dict[str, Any],
) -> dict[str, Any]:
    case_id = str(evidence_pack.get("case_id") or "")
    if not case_id:
        raise Stage5ArtifactError("Stage 5 requires a Stage 0 case_id")

    stage2_hash = _require_valid_artifact_hash(stage2_artifact, label="Stage 2")
    stage3_hash = _require_valid_artifact_hash(stage3_context_artifact, label="contextual Stage 3")
    stage4_hash = _require_valid_artifact_hash(stage4_context_artifact, label="contextual Stage 4")
    upstream_case_ids = {
        case_id,
        str(stage2_artifact.get("case_id") or ""),
        str(stage3_context_artifact.get("case_id") or ""),
        str(stage4_context_artifact.get("case_id") or ""),
    }
    if len(upstream_case_ids) != 1:
        raise Stage5ArtifactError("Stage 5 upstream artifact case_id mismatch")

    identity: dict[str, Any] = {
        "schema_version": STAGE5_ARTIFACT_SCHEMA_VERSION,
        "diagnostic_contract_version": STAGE5_DIAGNOSTIC_CONTRACT_VERSION,
        "case_id": case_id,
        "stage0_evidence_pack_sha256": _sha256_json(evidence_pack),
        "stage2_artifact_sha256": stage2_hash,
        "stage2_output_sha256": stage2_artifact.get("stage2_output_sha256"),
        "stage3_context_artifact_sha256": stage3_hash,
        "stage3_context_output_sha256": stage3_context_artifact.get("stage3_contextual_output_sha256"),
        "stage4_context_artifact_sha256": stage4_hash,
        "stage4_context_output_sha256": stage4_context_artifact.get("stage4_output_sha256"),
        "output_schema_sha256": _sha256_json(Stage5DiagnosticBundle.model_json_schema()),
    }
    if not identity["stage2_output_sha256"]:
        raise Stage5ArtifactError("Stage 5 requires Stage 2 output hash")
    if not identity["stage3_context_output_sha256"]:
        raise Stage5ArtifactError("Stage 5 requires contextual Stage 3 output hash")
    if not identity["stage4_context_output_sha256"]:
        raise Stage5ArtifactError("Stage 5 requires contextual Stage 4 output hash")
    identity["input_fingerprint"] = _sha256_json(identity)
    return identity


def stage5_artifact_path(*, project_root: str | Path, identity: dict[str, Any]) -> Path:
    return (
        Path(project_root).resolve()
        / "Multi-Agent"
        / "Results"
        / "Stage5"
        / "artifacts"
        / _safe(str(identity["case_id"]))
        / f"{identity['input_fingerprint']}.json"
    )


def load_frozen_stage5_artifact(
    *,
    project_root: str | Path,
    evidence_pack: dict[str, Any],
    stage2_artifact: dict[str, Any],
    stage3_context_artifact: dict[str, Any],
    stage4_context_artifact: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any], Path]:
    identity = build_stage5_identity(
        evidence_pack=evidence_pack,
        stage2_artifact=stage2_artifact,
        stage3_context_artifact=stage3_context_artifact,
        stage4_context_artifact=stage4_context_artifact,
    )
    path = stage5_artifact_path(project_root=project_root, identity=identity)
    if not path.exists():
        return None, identity, path
    artifact = json.loads(path.read_text(encoding="utf-8"))
    stored = artifact.get("artifact_sha256")
    unhashed = {key: value for key, value in artifact.items() if key != "artifact_sha256"}
    if not stored or stored != _sha256_json(unhashed):
        raise Stage5ArtifactError("Frozen Stage 5 artifact content hash mismatch")
    for key, expected in identity.items():
        if artifact.get(key) != expected:
            raise Stage5ArtifactError(f"Frozen Stage 5 identity mismatch for {key}")
    try:
        bundle = Stage5DiagnosticBundle.model_validate(artifact.get("diagnostic_bundle"))
    except Exception as exc:
        raise Stage5ArtifactError(f"Frozen Stage 5 diagnostic bundle invalid: {exc}") from exc
    output = bundle.model_dump(mode="json")
    if artifact.get("stage5_output_sha256") != _sha256_json(output):
        raise Stage5ArtifactError("Frozen Stage 5 output hash mismatch")
    return artifact, identity, path


def freeze_stage5_artifact(
    *,
    project_root: str | Path,
    evidence_pack: dict[str, Any],
    stage2_artifact: dict[str, Any],
    stage3_context_artifact: dict[str, Any],
    stage4_context_artifact: dict[str, Any],
    diagnostic_bundle: Stage5DiagnosticBundle | dict[str, Any],
) -> tuple[dict[str, Any], Path]:
    identity = build_stage5_identity(
        evidence_pack=evidence_pack,
        stage2_artifact=stage2_artifact,
        stage3_context_artifact=stage3_context_artifact,
        stage4_context_artifact=stage4_context_artifact,
    )
    try:
        bundle = (
            diagnostic_bundle
            if isinstance(diagnostic_bundle, Stage5DiagnosticBundle)
            else Stage5DiagnosticBundle.model_validate(diagnostic_bundle)
        )
    except Exception as exc:
        raise Stage5ValidationError(f"Invalid Stage 5 diagnostic bundle: {exc}") from exc
    if bundle.case_id != identity["case_id"]:
        raise Stage5ArtifactError("Stage 5 diagnostic case_id mismatch")
    output = bundle.model_dump(mode="json")
    artifact: dict[str, Any] = {
        **identity,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage5_output_sha256": _sha256_json(output),
        "diagnostic_bundle": output,
    }
    artifact["artifact_sha256"] = _sha256_json(artifact)
    path = stage5_artifact_path(project_root=project_root, identity=identity)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(artifact, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError:
        existing, _, _ = load_frozen_stage5_artifact(
            project_root=project_root,
            evidence_pack=evidence_pack,
            stage2_artifact=stage2_artifact,
            stage3_context_artifact=stage3_context_artifact,
            stage4_context_artifact=stage4_context_artifact,
        )
        if existing is None or existing.get("stage5_output_sha256") != artifact["stage5_output_sha256"]:
            raise Stage5ArtifactError("Conflicting frozen Stage 5 artifact for exact identity")
        return existing, path
    return artifact, path
