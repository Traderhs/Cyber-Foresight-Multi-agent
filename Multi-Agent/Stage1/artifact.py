from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from Stage1.runtime import get_experiment_runtime_profile
from Stage1.prompts import (
    ATTACK_FEASIBILITY_SYSTEM_PROMPT,
    DEFENSE_ROBUSTNESS_SYSTEM_PROMPT,
    STAGE1_EVALUATION_USER_PROMPT,
)
from Stage1.schema import CriticAssessment, CriticType, validate_critic_assessment


STAGE1_ARTIFACT_SCHEMA_VERSION = "stage1-artifact-v1"
STAGE1_CRITIC_CHECKPOINT_SCHEMA_VERSION = "stage1-critic-checkpoint-v1"


class Stage1ArtifactError(ValueError):
    """Raised when a frozen Stage 1 artifact violates its fixed contract."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_text(_canonical_json(value))


def _safe_case_id(case_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", case_id).strip("._")
    return safe or "case"


def build_stage1_identity(*, evidence_pack: dict[str, Any], forecast_data: str) -> dict[str, Any]:
    """Build the identity that determines whether a Stage 1 result may be reused."""
    case_id = str(evidence_pack.get("case_id") or "")
    if not case_id:
        raise Stage1ArtifactError("Stage 0 EvidencePack is missing case_id")

    identity: dict[str, Any] = {
        "schema_version": STAGE1_ARTIFACT_SCHEMA_VERSION,
        "case_id": case_id,
        "stage0_evidence_pack_sha256": _sha256_json(evidence_pack),
        "forecast_data_sha256": _sha256_text(forecast_data),
        "prompt_hashes": {
            "attack_feasibility_system_prompt_sha256": _sha256_text(ATTACK_FEASIBILITY_SYSTEM_PROMPT),
            "defense_robustness_system_prompt_sha256": _sha256_text(DEFENSE_ROBUSTNESS_SYSTEM_PROMPT),
            "stage1_user_prompt_sha256": _sha256_text(STAGE1_EVALUATION_USER_PROMPT),
        },
        "output_schema_sha256": _sha256_json(CriticAssessment.model_json_schema()),
        "runtime": get_experiment_runtime_profile(),
    }
    identity["input_fingerprint"] = _sha256_json(identity)
    return identity


def stage1_artifact_path(*, project_root: str | Path, identity: dict[str, Any]) -> Path:
    root = Path(project_root).resolve()
    return (
        root
        / "Data"
        / "Stage1"
        / "artifacts"
        / _safe_case_id(str(identity["case_id"]))
        / f"{identity['input_fingerprint']}.json"
    )


def stage1_critic_checkpoint_path(
    *,
    project_root: str | Path,
    identity: dict[str, Any],
    critic_type: CriticType,
) -> Path:
    root = Path(project_root).resolve()
    return (
        root
        / "Data"
        / "Stage1"
        / "checkpoints"
        / _safe_case_id(str(identity["case_id"]))
        / str(identity["input_fingerprint"])
        / f"{critic_type.value}.json"
    )


def _validate_critic_checkpoint(
    checkpoint: dict[str, Any],
    *,
    expected_identity: dict[str, Any],
    evidence_pack: dict[str, Any],
    critic_type: CriticType,
) -> dict[str, Any]:
    stored_hash = checkpoint.get("checkpoint_sha256")
    if not stored_hash:
        raise Stage1ArtifactError("Stage 1 critic checkpoint is missing checkpoint_sha256")
    unhashed = {key: value for key, value in checkpoint.items() if key != "checkpoint_sha256"}
    actual_hash = _sha256_json(unhashed)
    if actual_hash != stored_hash:
        raise Stage1ArtifactError(
            f"Stage 1 critic checkpoint hash mismatch: stored={stored_hash}, actual={actual_hash}"
        )
    if checkpoint.get("schema_version") != STAGE1_CRITIC_CHECKPOINT_SCHEMA_VERSION:
        raise Stage1ArtifactError("Stage 1 critic checkpoint schema version mismatch")
    if checkpoint.get("case_id") != expected_identity["case_id"]:
        raise Stage1ArtifactError("Stage 1 critic checkpoint case_id mismatch")
    if checkpoint.get("input_fingerprint") != expected_identity["input_fingerprint"]:
        raise Stage1ArtifactError("Stage 1 critic checkpoint fingerprint mismatch")
    if checkpoint.get("critic_type") != critic_type.value:
        raise Stage1ArtifactError("Stage 1 critic checkpoint critic_type mismatch")

    assessment = validate_critic_assessment(
        checkpoint.get("assessment"),
        evidence_pack=evidence_pack,
        expected_critic_type=critic_type,
    )
    expected_assessment_hash = _sha256_json(assessment.model_dump(mode="json"))
    if checkpoint.get("assessment_sha256") != expected_assessment_hash:
        raise Stage1ArtifactError("Stage 1 critic checkpoint assessment hash mismatch")
    return checkpoint


def load_stage1_critic_checkpoint(
    *,
    project_root: str | Path,
    evidence_pack: dict[str, Any],
    forecast_data: str,
    critic_type: CriticType,
) -> tuple[dict[str, Any] | None, Path]:
    """Load only an exact-fingerprint per-critic checkpoint; never fall back."""
    identity = build_stage1_identity(evidence_pack=evidence_pack, forecast_data=forecast_data)
    path = stage1_critic_checkpoint_path(
        project_root=project_root,
        identity=identity,
        critic_type=critic_type,
    )
    if not path.exists():
        return None, path
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    return (
        _validate_critic_checkpoint(
            checkpoint,
            expected_identity=identity,
            evidence_pack=evidence_pack,
            critic_type=critic_type,
        ),
        path,
    )


def freeze_stage1_critic_checkpoint(
    *,
    project_root: str | Path,
    evidence_pack: dict[str, Any],
    forecast_data: str,
    critic_type: CriticType,
    assessment: CriticAssessment | dict[str, Any],
) -> Path:
    """Persist one validated critic output immediately so interrupted runs can resume."""
    identity = build_stage1_identity(evidence_pack=evidence_pack, forecast_data=forecast_data)
    parsed = validate_critic_assessment(
        assessment,
        evidence_pack=evidence_pack,
        expected_critic_type=critic_type,
    )
    assessment_dict = parsed.model_dump(mode="json")
    checkpoint: dict[str, Any] = {
        "schema_version": STAGE1_CRITIC_CHECKPOINT_SCHEMA_VERSION,
        "case_id": identity["case_id"],
        "input_fingerprint": identity["input_fingerprint"],
        "critic_type": critic_type.value,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "assessment_sha256": _sha256_json(assessment_dict),
        "assessment": assessment_dict,
    }
    checkpoint["checkpoint_sha256"] = _sha256_json(checkpoint)
    path = stage1_critic_checkpoint_path(
        project_root=project_root,
        identity=identity,
        critic_type=critic_type,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(checkpoint, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError:
        existing = json.loads(path.read_text(encoding="utf-8"))
        _validate_critic_checkpoint(
            existing,
            expected_identity=identity,
            evidence_pack=evidence_pack,
            critic_type=critic_type,
        )
        if existing.get("assessment_sha256") != checkpoint["assessment_sha256"]:
            raise Stage1ArtifactError(
                "A Stage 1 critic checkpoint already exists for this exact experiment identity "
                "with a different assessment; refusing to overwrite it."
            )
    return path


def clear_stage1_critic_checkpoints(
    *,
    project_root: str | Path,
    evidence_pack: dict[str, Any],
    forecast_data: str,
) -> None:
    """Remove redundant partial checkpoints only after the final immutable artifact exists."""
    identity = build_stage1_identity(evidence_pack=evidence_pack, forecast_data=forecast_data)
    for critic_type in CriticType:
        path = stage1_critic_checkpoint_path(
            project_root=project_root,
            identity=identity,
            critic_type=critic_type,
        )
        if path.exists():
            path.unlink()
    checkpoint_dir = (
        Path(project_root).resolve()
        / "Data"
        / "Stage1"
        / "checkpoints"
        / _safe_case_id(str(identity["case_id"]))
        / str(identity["input_fingerprint"])
    )
    try:
        checkpoint_dir.rmdir()
    except OSError:
        pass


def _validate_frozen_artifact(
    artifact: dict[str, Any],
    *,
    expected_identity: dict[str, Any],
    evidence_pack: dict[str, Any],
) -> dict[str, Any]:
    stored_hash = artifact.get("artifact_sha256")
    if not stored_hash:
        raise Stage1ArtifactError("Frozen Stage 1 artifact is missing artifact_sha256")
    unhashed = {key: value for key, value in artifact.items() if key != "artifact_sha256"}
    actual_hash = _sha256_json(unhashed)
    if actual_hash != stored_hash:
        raise Stage1ArtifactError(
            f"Frozen Stage 1 artifact hash mismatch: stored={stored_hash}, actual={actual_hash}"
        )

    for key, expected in expected_identity.items():
        if artifact.get(key) != expected:
            raise Stage1ArtifactError(f"Frozen Stage 1 artifact identity mismatch for {key}")

    assessments = artifact.get("assessments") or {}
    attack = validate_critic_assessment(
        assessments.get("attack_feasibility"),
        evidence_pack=evidence_pack,
        expected_critic_type=CriticType.ATTACK_FEASIBILITY,
    )
    defense = validate_critic_assessment(
        assessments.get("defense_robustness"),
        evidence_pack=evidence_pack,
        expected_critic_type=CriticType.DEFENSE_ROBUSTNESS,
    )
    expected_output_hash = _sha256_json(
        {
            "attack_feasibility": attack.model_dump(mode="json"),
            "defense_robustness": defense.model_dump(mode="json"),
        }
    )
    if artifact.get("stage1_output_sha256") != expected_output_hash:
        raise Stage1ArtifactError("Frozen Stage 1 output hash mismatch")
    return artifact


def load_frozen_stage1_artifact(
    *, project_root: str | Path, evidence_pack: dict[str, Any], forecast_data: str
) -> tuple[dict[str, Any] | None, dict[str, Any], Path]:
    """Load only the exact matching artifact; never fall back to another run/version."""
    identity = build_stage1_identity(evidence_pack=evidence_pack, forecast_data=forecast_data)
    path = stage1_artifact_path(project_root=project_root, identity=identity)
    if not path.exists():
        return None, identity, path
    artifact = json.loads(path.read_text(encoding="utf-8"))
    return (
        _validate_frozen_artifact(
            artifact,
            expected_identity=identity,
            evidence_pack=evidence_pack,
        ),
        identity,
        path,
    )


def freeze_stage1_artifact(
    *,
    project_root: str | Path,
    evidence_pack: dict[str, Any],
    forecast_data: str,
    attack_assessment: CriticAssessment | dict[str, Any],
    defense_assessment: CriticAssessment | dict[str, Any],
) -> tuple[dict[str, Any], Path]:
    """Create an immutable content-addressed Stage 1 artifact exactly once."""
    identity = build_stage1_identity(evidence_pack=evidence_pack, forecast_data=forecast_data)
    path = stage1_artifact_path(project_root=project_root, identity=identity)

    attack = validate_critic_assessment(
        attack_assessment,
        evidence_pack=evidence_pack,
        expected_critic_type=CriticType.ATTACK_FEASIBILITY,
    )
    defense = validate_critic_assessment(
        defense_assessment,
        evidence_pack=evidence_pack,
        expected_critic_type=CriticType.DEFENSE_ROBUSTNESS,
    )
    outputs = {
        "attack_feasibility": attack.model_dump(mode="json"),
        "defense_robustness": defense.model_dump(mode="json"),
    }
    artifact: dict[str, Any] = {
        **identity,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_snapshot_id": evidence_pack.get("source_snapshot_id"),
        "stage1_output_sha256": _sha256_json(outputs),
        "assessments": outputs,
    }
    artifact["artifact_sha256"] = _sha256_json(artifact)

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(artifact, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError:
        existing = json.loads(path.read_text(encoding="utf-8"))
        existing = _validate_frozen_artifact(
            existing,
            expected_identity=identity,
            evidence_pack=evidence_pack,
        )
        if existing.get("stage1_output_sha256") != artifact["stage1_output_sha256"]:
            raise Stage1ArtifactError(
                "A frozen Stage 1 artifact already exists for this exact experiment identity "
                "with different outputs; refusing to overwrite it."
            )
        return existing, path

    return artifact, path
