from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from Stage1.runtime import get_experiment_runtime_profile
from Stage3.artifact import STAGE3_ARTIFACT_SCHEMA_VERSION
from Stage3.schema import DecisionObject, Stage3DecisionBundle
from Stage3.schema import STAGE3_SEMANTIC_VALIDATION_VERSION
from Stage4.input import build_stage4_evaluation_payload
from Stage4.prompts import STAGE4_EVALUATION_USER_PROMPT, SYSTEM_PROMPT_BY_LENS
from Stage4.schema import (
    STAGE4_SEMANTIC_VALIDATION_VERSION,
    LensAssessment,
    LensType,
    Stage4EvaluationBundle,
    build_constrained_lens_response_schema,
    validate_lens_assessment,
    validate_stage4_evaluation_bundle,
)


STAGE4_ARTIFACT_SCHEMA_VERSION = "stage4-artifact-v1"
STAGE4_LENS_CHECKPOINT_SCHEMA_VERSION = "stage4-lens-checkpoint-v1"


class Stage4ArtifactError(ValueError):
    """Raised when a Stage 4 checkpoint/artifact violates its content-addressed contract."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_text(_canonical_json(value))


def _safe(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return safe or "item"


def _validate_stage3_artifact_binding(
    *,
    stage3_artifact: dict[str, Any],
    evidence_pack: dict[str, Any],
) -> Stage3DecisionBundle:
    stored_hash = str(stage3_artifact.get("artifact_sha256") or "")
    output_hash = str(stage3_artifact.get("stage3_output_sha256") or "")
    input_fingerprint = str(stage3_artifact.get("input_fingerprint") or "")
    raw_bundle = stage3_artifact.get("decision_bundle")
    if not stored_hash or not output_hash or not input_fingerprint or raw_bundle is None:
        raise Stage4ArtifactError("Stage 4 requires a complete frozen Stage 3 artifact")
    if stage3_artifact.get("schema_version") != STAGE3_ARTIFACT_SCHEMA_VERSION:
        raise Stage4ArtifactError("Stage 4 rejected a Stage 3 artifact with an incompatible schema version")
    if stage3_artifact.get("semantic_validation_version") != STAGE3_SEMANTIC_VALIDATION_VERSION:
        raise Stage4ArtifactError(
            "Stage 4 rejected a Stage 3 artifact with an incompatible semantic validation version"
        )
    unhashed = {key: value for key, value in stage3_artifact.items() if key != "artifact_sha256"}
    if stored_hash != _sha256_json(unhashed):
        raise Stage4ArtifactError("Stage 4 rejected a Stage 3 artifact with an invalid content hash")
    expected_stage0_hash = _sha256_json(evidence_pack)
    if stage3_artifact.get("stage0_evidence_pack_sha256") != expected_stage0_hash:
        raise Stage4ArtifactError(
            "Stage 4 rejected a Stage 3 artifact that is not bound to the current Stage 0 EvidencePack"
        )
    bundle = Stage3DecisionBundle.model_validate(raw_bundle)
    bundle_dict = bundle.model_dump(mode="json")
    if output_hash != _sha256_json(bundle_dict):
        raise Stage4ArtifactError("Stage 4 rejected a Stage 3 artifact with an invalid output hash")
    expected_case_id = str(evidence_pack.get("case_id") or "")
    if bundle.case_id != expected_case_id:
        raise Stage4ArtifactError("Stage 3/Stage 0 case_id mismatch at Stage 4 boundary")
    pack_ids = {
        str(record.get("evidence_id"))
        for record in evidence_pack.get("evidence", [])
        if record.get("evidence_id")
    }
    if not pack_ids:
        raise Stage4ArtifactError("Stage 4 requires a non-empty frozen Stage 0 evidence set")
    for decision_object in bundle.decision_objects:
        if set(decision_object.evaluation_evidence_ids) != pack_ids:
            raise Stage4ArtifactError(
                "Stage 4 requires every frozen DecisionObject to carry the exact Stage 0 evidence universe"
            )
    return bundle


def build_stage4_identity(
    *,
    evidence_pack: dict[str, Any],
    stage3_artifact: dict[str, Any],
) -> dict[str, Any]:
    bundle = _validate_stage3_artifact_binding(
        stage3_artifact=stage3_artifact,
        evidence_pack=evidence_pack,
    )
    prompt_hashes = {
        f"{lens.value}_system_prompt_sha256": _sha256_text(SYSTEM_PROMPT_BY_LENS[lens.value])
        for lens in LensType
    }
    prompt_hashes["stage4_user_prompt_sha256"] = _sha256_text(STAGE4_EVALUATION_USER_PROMPT)
    constrained_schema_hashes: dict[str, dict[str, str]] = {}
    payload_hashes: dict[str, str] = {}
    for decision_object in bundle.decision_objects:
        payload = build_stage4_evaluation_payload(
            decision_object=decision_object,
            evidence_pack=evidence_pack,
        )
        payload_hashes[decision_object.decision_object_id] = _sha256_json(payload)
        constrained_schema_hashes[decision_object.decision_object_id] = {
            lens.value: _sha256_json(
                build_constrained_lens_response_schema(
                    decision_object=decision_object,
                    evidence_pack=evidence_pack,
                    expected_lens_type=lens,
                )
            )
            for lens in LensType
        }

    identity: dict[str, Any] = {
        "schema_version": STAGE4_ARTIFACT_SCHEMA_VERSION,
        "semantic_validation_version": STAGE4_SEMANTIC_VALIDATION_VERSION,
        "case_id": bundle.case_id,
        "stage0_evidence_pack_sha256": _sha256_json(evidence_pack),
        "stage3_artifact_sha256": stage3_artifact["artifact_sha256"],
        "stage3_output_sha256": stage3_artifact["stage3_output_sha256"],
        "stage3_input_fingerprint": stage3_artifact.get("input_fingerprint"),
        "decision_object_payload_sha256": payload_hashes,
        "prompt_hashes": prompt_hashes,
        "output_schema_sha256": _sha256_json(Stage4EvaluationBundle.model_json_schema()),
        "lens_schema_sha256": _sha256_json(LensAssessment.model_json_schema()),
        "constrained_output_schema_sha256": constrained_schema_hashes,
        "runtime": get_experiment_runtime_profile(),
    }
    identity["input_fingerprint"] = _sha256_json(identity)
    return identity


def stage4_artifact_path(*, project_root: str | Path, identity: dict[str, Any]) -> Path:
    return (
        Path(project_root).resolve()
        / "Multi-Agent"
        / "Results"
        / "Stage4"
        / "artifacts"
        / _safe(str(identity["case_id"]))
        / f"{identity['input_fingerprint']}.json"
    )


def stage4_lens_checkpoint_path(
    *,
    project_root: str | Path,
    identity: dict[str, Any],
    decision_object_id: str,
    lens_type: LensType,
) -> Path:
    return (
        Path(project_root).resolve()
        / "Multi-Agent"
        / "Results"
        / "Stage4"
        / "checkpoints"
        / _safe(str(identity["case_id"]))
        / str(identity["input_fingerprint"])
        / _safe(decision_object_id)
        / f"{lens_type.value}.json"
    )


def _decision_object_by_id(bundle: Stage3DecisionBundle, decision_object_id: str) -> DecisionObject:
    matches = [item for item in bundle.decision_objects if item.decision_object_id == decision_object_id]
    if len(matches) != 1:
        raise Stage4ArtifactError(f"Unknown or duplicate Stage 3 decision_object_id: {decision_object_id!r}")
    return matches[0]


def load_stage4_lens_checkpoint(
    *,
    project_root: str | Path,
    evidence_pack: dict[str, Any],
    stage3_artifact: dict[str, Any],
    decision_object_id: str,
    lens_type: LensType,
) -> tuple[dict[str, Any] | None, Path]:
    identity = build_stage4_identity(evidence_pack=evidence_pack, stage3_artifact=stage3_artifact)
    bundle = _validate_stage3_artifact_binding(stage3_artifact=stage3_artifact, evidence_pack=evidence_pack)
    decision_object = _decision_object_by_id(bundle, decision_object_id)
    path = stage4_lens_checkpoint_path(
        project_root=project_root,
        identity=identity,
        decision_object_id=decision_object_id,
        lens_type=lens_type,
    )
    if not path.exists():
        return None, path
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    stored_hash = checkpoint.get("checkpoint_sha256")
    unhashed = {key: value for key, value in checkpoint.items() if key != "checkpoint_sha256"}
    if not stored_hash or stored_hash != _sha256_json(unhashed):
        raise Stage4ArtifactError("Stage 4 lens checkpoint hash mismatch")
    if checkpoint.get("schema_version") != STAGE4_LENS_CHECKPOINT_SCHEMA_VERSION:
        raise Stage4ArtifactError("Stage 4 lens checkpoint schema version mismatch")
    if checkpoint.get("input_fingerprint") != identity["input_fingerprint"]:
        raise Stage4ArtifactError("Stage 4 lens checkpoint fingerprint mismatch")
    if checkpoint.get("decision_object_id") != decision_object_id:
        raise Stage4ArtifactError("Stage 4 lens checkpoint decision_object_id mismatch")
    if checkpoint.get("lens_type") != lens_type.value:
        raise Stage4ArtifactError("Stage 4 lens checkpoint lens_type mismatch")
    assessment = validate_lens_assessment(
        checkpoint.get("assessment"),
        decision_object=decision_object,
        evidence_pack=evidence_pack,
        expected_lens_type=lens_type,
    )
    if checkpoint.get("assessment_sha256") != _sha256_json(assessment.model_dump(mode="json")):
        raise Stage4ArtifactError("Stage 4 lens checkpoint assessment hash mismatch")
    return checkpoint, path


def freeze_stage4_lens_checkpoint(
    *,
    project_root: str | Path,
    evidence_pack: dict[str, Any],
    stage3_artifact: dict[str, Any],
    decision_object_id: str,
    lens_type: LensType,
    assessment: LensAssessment | dict[str, Any],
) -> Path:
    identity = build_stage4_identity(evidence_pack=evidence_pack, stage3_artifact=stage3_artifact)
    bundle = _validate_stage3_artifact_binding(stage3_artifact=stage3_artifact, evidence_pack=evidence_pack)
    decision_object = _decision_object_by_id(bundle, decision_object_id)
    parsed = validate_lens_assessment(
        assessment,
        decision_object=decision_object,
        evidence_pack=evidence_pack,
        expected_lens_type=lens_type,
    )
    output = parsed.model_dump(mode="json")
    checkpoint: dict[str, Any] = {
        "schema_version": STAGE4_LENS_CHECKPOINT_SCHEMA_VERSION,
        "case_id": identity["case_id"],
        "input_fingerprint": identity["input_fingerprint"],
        "decision_object_id": decision_object_id,
        "lens_type": lens_type.value,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "assessment_sha256": _sha256_json(output),
        "assessment": output,
    }
    checkpoint["checkpoint_sha256"] = _sha256_json(checkpoint)
    path = stage4_lens_checkpoint_path(
        project_root=project_root,
        identity=identity,
        decision_object_id=decision_object_id,
        lens_type=lens_type,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(checkpoint, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError:
        existing, _ = load_stage4_lens_checkpoint(
            project_root=project_root,
            evidence_pack=evidence_pack,
            stage3_artifact=stage3_artifact,
            decision_object_id=decision_object_id,
            lens_type=lens_type,
        )
        if existing is None or existing.get("assessment_sha256") != checkpoint["assessment_sha256"]:
            raise Stage4ArtifactError(
                "A Stage 4 lens checkpoint already exists for this exact identity with a different assessment"
            )
    return path


def clear_stage4_lens_checkpoints(
    *,
    project_root: str | Path,
    evidence_pack: dict[str, Any],
    stage3_artifact: dict[str, Any],
) -> None:
    identity = build_stage4_identity(evidence_pack=evidence_pack, stage3_artifact=stage3_artifact)
    bundle = _validate_stage3_artifact_binding(stage3_artifact=stage3_artifact, evidence_pack=evidence_pack)
    for decision_object in bundle.decision_objects:
        for lens_type in LensType:
            path = stage4_lens_checkpoint_path(
                project_root=project_root,
                identity=identity,
                decision_object_id=decision_object.decision_object_id,
                lens_type=lens_type,
            )
            if path.exists():
                path.unlink()
            try:
                path.parent.rmdir()
            except OSError:
                pass
    checkpoint_root = (
        Path(project_root).resolve()
        / "Multi-Agent"
        / "Results"
        / "Stage4"
        / "checkpoints"
        / _safe(str(identity["case_id"]))
        / str(identity["input_fingerprint"])
    )
    try:
        checkpoint_root.rmdir()
    except OSError:
        pass


def load_frozen_stage4_artifact(
    *,
    project_root: str | Path,
    evidence_pack: dict[str, Any],
    stage3_artifact: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any], Path]:
    identity = build_stage4_identity(evidence_pack=evidence_pack, stage3_artifact=stage3_artifact)
    path = stage4_artifact_path(project_root=project_root, identity=identity)
    if not path.exists():
        return None, identity, path
    artifact = json.loads(path.read_text(encoding="utf-8"))
    stored_hash = artifact.get("artifact_sha256")
    unhashed = {key: value for key, value in artifact.items() if key != "artifact_sha256"}
    if not stored_hash or stored_hash != _sha256_json(unhashed):
        raise Stage4ArtifactError("Frozen Stage 4 artifact hash mismatch")
    for key, expected in identity.items():
        if artifact.get(key) != expected:
            raise Stage4ArtifactError(f"Frozen Stage 4 artifact identity mismatch for {key}")
    stage3_bundle = _validate_stage3_artifact_binding(
        stage3_artifact=stage3_artifact,
        evidence_pack=evidence_pack,
    )
    bundle = validate_stage4_evaluation_bundle(
        artifact.get("evaluation_bundle"),
        decision_objects=stage3_bundle.decision_objects,
        evidence_pack=evidence_pack,
    )
    if artifact.get("stage4_output_sha256") != _sha256_json(bundle.model_dump(mode="json")):
        raise Stage4ArtifactError("Frozen Stage 4 output hash mismatch")
    return artifact, identity, path


def freeze_stage4_artifact(
    *,
    project_root: str | Path,
    evidence_pack: dict[str, Any],
    stage3_artifact: dict[str, Any],
    evaluation_bundle: Stage4EvaluationBundle | dict[str, Any],
) -> tuple[dict[str, Any], Path]:
    identity = build_stage4_identity(evidence_pack=evidence_pack, stage3_artifact=stage3_artifact)
    stage3_bundle = _validate_stage3_artifact_binding(
        stage3_artifact=stage3_artifact,
        evidence_pack=evidence_pack,
    )
    bundle = validate_stage4_evaluation_bundle(
        evaluation_bundle,
        decision_objects=stage3_bundle.decision_objects,
        evidence_pack=evidence_pack,
    )
    output = bundle.model_dump(mode="json")
    artifact: dict[str, Any] = {
        **identity,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_snapshot_id": evidence_pack.get("source_snapshot_id"),
        "stage4_output_sha256": _sha256_json(output),
        "evaluation_bundle": output,
    }
    artifact["artifact_sha256"] = _sha256_json(artifact)
    path = stage4_artifact_path(project_root=project_root, identity=identity)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(artifact, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError:
        existing, _, _ = load_frozen_stage4_artifact(
            project_root=project_root,
            evidence_pack=evidence_pack,
            stage3_artifact=stage3_artifact,
        )
        if existing is None or existing.get("stage4_output_sha256") != artifact["stage4_output_sha256"]:
            raise Stage4ArtifactError(
                "A frozen Stage 4 artifact already exists for this exact identity with different outputs"
            )
        return existing, path
    return artifact, path

