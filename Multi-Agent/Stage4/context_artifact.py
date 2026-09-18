from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from Stage3.context_schema import ContextualDecisionObject, Stage3ContextualDecisionBundle
from Stage3.context_artifact import STAGE3_CONTEXT_ARTIFACT_SCHEMA_VERSION
from Stage3.context_schema import STAGE3_CONTEXT_SEMANTIC_VALIDATION_VERSION
from Stage0.context_scenarios import CONTEXT_SCENARIO_SET_VERSION
from Stage4.context_input import build_context_stage4_evaluation_payload
from Stage4.context_prompts import CONTEXT_STAGE4_EVALUATION_USER_PROMPT, CONTEXT_SYSTEM_PROMPT_BY_LENS
from Stage4.context_schema import (
    STAGE4_CONTEXT_SEMANTIC_VALIDATION_VERSION,
    LensAssessment,
    LensType,
    Stage4EvaluationBundle,
    build_constrained_context_lens_response_schema,
    validate_context_lens_assessment,
    validate_context_stage4_bundle,
)
from Stage4.context_runtime import get_stage4_context_runtime_profile


STAGE4_CONTEXT_ARTIFACT_SCHEMA_VERSION = "stage4-artifact-v4"
STAGE4_CONTEXT_LENS_CHECKPOINT_SCHEMA_VERSION = "stage4-lens-checkpoint-v4"
STAGE4_CONTEXT_GENERATION_CONTRACT_VERSION = "jurisdiction-neutral-equivalence-dedup-v1"


class Stage4ContextArtifactError(ValueError):
    pass


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_text(_canonical_json(value))


def _safe(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return safe or "item"


def _validate_stage3_context_artifact(stage3_context_artifact: dict[str, Any]) -> Stage3ContextualDecisionBundle:
    if stage3_context_artifact.get("schema_version") != STAGE3_CONTEXT_ARTIFACT_SCHEMA_VERSION:
        raise Stage4ContextArtifactError("Stage 4 v4 requires the current contextual Stage 3 artifact schema")
    if stage3_context_artifact.get("semantic_validation_version") != STAGE3_CONTEXT_SEMANTIC_VALIDATION_VERSION:
        raise Stage4ContextArtifactError("Stage 4 v4 requires the current contextual Stage 3 semantic contract")
    if stage3_context_artifact.get("scenario_set_version") != CONTEXT_SCENARIO_SET_VERSION:
        raise Stage4ContextArtifactError("Stage 4 v4 requires the current contextual scenario set")
    stored = str(stage3_context_artifact.get("artifact_sha256") or "")
    output_hash = str(stage3_context_artifact.get("stage3_contextual_output_sha256") or "")
    raw_bundle = stage3_context_artifact.get("contextual_decision_bundle")
    if not stored or not output_hash or raw_bundle is None:
        raise Stage4ContextArtifactError("Stage 4 v4 requires a complete contextual Stage 3 artifact")
    unhashed = {key: value for key, value in stage3_context_artifact.items() if key != "artifact_sha256"}
    if stored != _sha256_json(unhashed):
        raise Stage4ContextArtifactError("Contextual Stage 3 artifact content hash mismatch")
    bundle = Stage3ContextualDecisionBundle.model_validate(raw_bundle)
    if output_hash != _sha256_json(bundle.model_dump(mode="json")):
        raise Stage4ContextArtifactError("Contextual Stage 3 artifact output hash mismatch")
    return bundle


def build_stage4_context_identity(*, stage3_context_artifact: dict[str, Any]) -> dict[str, Any]:
    bundle = _validate_stage3_context_artifact(stage3_context_artifact)
    prompt_hashes = {
        f"{lens.value}_system_prompt_sha256": _sha256_text(CONTEXT_SYSTEM_PROMPT_BY_LENS[lens.value])
        for lens in LensType
    }
    prompt_hashes["stage4_user_prompt_sha256"] = _sha256_text(CONTEXT_STAGE4_EVALUATION_USER_PROMPT)
    payload_hashes: dict[str, dict[str, str]] = {}
    constrained_schema_hashes: dict[str, dict[str, str]] = {}
    for obj in bundle.contextual_decision_objects:
        payload_hashes[obj.decision_object_id] = {
            lens.value: _sha256_json(
                build_context_stage4_evaluation_payload(
                    decision_object=obj,
                    expected_lens_type=lens,
                )
            )
            for lens in LensType
        }
        constrained_schema_hashes[obj.decision_object_id] = {
            lens.value: _sha256_json(
                build_constrained_context_lens_response_schema(
                    decision_object=obj,
                    expected_lens_type=lens,
                )
            )
            for lens in LensType
        }
    identity: dict[str, Any] = {
        "schema_version": STAGE4_CONTEXT_ARTIFACT_SCHEMA_VERSION,
        "semantic_validation_version": STAGE4_CONTEXT_SEMANTIC_VALIDATION_VERSION,
        "generation_contract_version": STAGE4_CONTEXT_GENERATION_CONTRACT_VERSION,
        "case_id": bundle.case_id,
        "scenario_set_version": bundle.scenario_set_version,
        "stage3_context_artifact_sha256": stage3_context_artifact["artifact_sha256"],
        "stage3_context_output_sha256": stage3_context_artifact["stage3_contextual_output_sha256"],
        "stage3_context_input_fingerprint": stage3_context_artifact.get("input_fingerprint"),
        "decision_object_payload_sha256": payload_hashes,
        "prompt_hashes": prompt_hashes,
        "output_schema_sha256": _sha256_json(Stage4EvaluationBundle.model_json_schema()),
        "lens_schema_sha256": _sha256_json(LensAssessment.model_json_schema()),
        "constrained_output_schema_sha256": constrained_schema_hashes,
        "runtime": get_stage4_context_runtime_profile(),
    }
    identity["input_fingerprint"] = _sha256_json(identity)
    return identity


def stage4_context_artifact_path(*, project_root: str | Path, identity: dict[str, Any]) -> Path:
    return (
        Path(project_root).resolve()
        / "Multi-Agent"
        / "Results"
        / "Stage4"
        / "contextual_artifacts"
        / _safe(str(identity["case_id"]))
        / f"{identity['input_fingerprint']}.json"
    )


def stage4_context_checkpoint_path(
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
        / "contextual_checkpoints"
        / _safe(str(identity["case_id"]))
        / str(identity["input_fingerprint"])
        / _safe(decision_object_id)
        / f"{lens_type.value}.json"
    )


def _object_by_id(bundle: Stage3ContextualDecisionBundle, decision_object_id: str) -> ContextualDecisionObject:
    matches = [item for item in bundle.contextual_decision_objects if item.decision_object_id == decision_object_id]
    if len(matches) != 1:
        raise Stage4ContextArtifactError(f"Unknown contextual decision_object_id: {decision_object_id!r}")
    return matches[0]


def load_stage4_context_lens_checkpoint(
    *,
    project_root: str | Path,
    stage3_context_artifact: dict[str, Any],
    decision_object_id: str,
    lens_type: LensType,
) -> tuple[dict[str, Any] | None, Path]:
    identity = build_stage4_context_identity(stage3_context_artifact=stage3_context_artifact)
    bundle = _validate_stage3_context_artifact(stage3_context_artifact)
    obj = _object_by_id(bundle, decision_object_id)
    path = stage4_context_checkpoint_path(
        project_root=project_root,
        identity=identity,
        decision_object_id=decision_object_id,
        lens_type=lens_type,
    )
    if not path.exists():
        return None, path
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    stored = checkpoint.get("checkpoint_sha256")
    unhashed = {key: value for key, value in checkpoint.items() if key != "checkpoint_sha256"}
    if not stored or stored != _sha256_json(unhashed):
        raise Stage4ContextArtifactError("Contextual Stage 4 checkpoint hash mismatch")
    if checkpoint.get("schema_version") != STAGE4_CONTEXT_LENS_CHECKPOINT_SCHEMA_VERSION:
        raise Stage4ContextArtifactError("Contextual Stage 4 checkpoint schema mismatch")
    if checkpoint.get("input_fingerprint") != identity["input_fingerprint"]:
        raise Stage4ContextArtifactError("Contextual Stage 4 checkpoint fingerprint mismatch")
    if checkpoint.get("decision_object_id") != decision_object_id or checkpoint.get("lens_type") != lens_type.value:
        raise Stage4ContextArtifactError("Contextual Stage 4 checkpoint identity mismatch")
    assessment = validate_context_lens_assessment(
        checkpoint.get("assessment"),
        decision_object=obj,
        expected_lens_type=lens_type,
    )
    if checkpoint.get("assessment_sha256") != _sha256_json(assessment.model_dump(mode="json")):
        raise Stage4ContextArtifactError("Contextual Stage 4 checkpoint assessment hash mismatch")
    return checkpoint, path


def freeze_stage4_context_lens_checkpoint(
    *,
    project_root: str | Path,
    stage3_context_artifact: dict[str, Any],
    decision_object_id: str,
    lens_type: LensType,
    assessment: LensAssessment | dict[str, Any],
) -> Path:
    identity = build_stage4_context_identity(stage3_context_artifact=stage3_context_artifact)
    bundle = _validate_stage3_context_artifact(stage3_context_artifact)
    obj = _object_by_id(bundle, decision_object_id)
    parsed = validate_context_lens_assessment(
        assessment,
        decision_object=obj,
        expected_lens_type=lens_type,
    )
    output = parsed.model_dump(mode="json")
    checkpoint: dict[str, Any] = {
        "schema_version": STAGE4_CONTEXT_LENS_CHECKPOINT_SCHEMA_VERSION,
        "case_id": identity["case_id"],
        "scenario_id": obj.scenario_id,
        "input_fingerprint": identity["input_fingerprint"],
        "decision_object_id": decision_object_id,
        "lens_type": lens_type.value,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "assessment_sha256": _sha256_json(output),
        "assessment": output,
    }
    checkpoint["checkpoint_sha256"] = _sha256_json(checkpoint)
    path = stage4_context_checkpoint_path(
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
        existing, _ = load_stage4_context_lens_checkpoint(
            project_root=project_root,
            stage3_context_artifact=stage3_context_artifact,
            decision_object_id=decision_object_id,
            lens_type=lens_type,
        )
        if existing is None or existing.get("assessment_sha256") != checkpoint["assessment_sha256"]:
            raise Stage4ContextArtifactError("Conflicting contextual Stage 4 checkpoint for exact identity")
    return path


def replace_stage4_context_lens_checkpoint_after_semantic_repair(
    *,
    project_root: str | Path,
    stage3_context_artifact: dict[str, Any],
    decision_object_id: str,
    lens_type: LensType,
    previous_assessment_sha256: str,
    assessment: LensAssessment | dict[str, Any],
    repair_reason: str,
) -> Path:
    """Replace one exact checkpoint only after a bounded semantic repair.

    Per-lens checkpoints are resumability aids rather than final research
    artifacts. A bundle-level semantic rule can therefore invalidate one
    otherwise well-formed checkpoint. Replacement is allowed only when the
    exact prior checkpoint still exists, passes integrity validation, and its
    assessment hash matches the caller's expected value. This prevents the
    repair path from becoming a general overwrite mechanism.
    """

    identity = build_stage4_context_identity(stage3_context_artifact=stage3_context_artifact)
    bundle = _validate_stage3_context_artifact(stage3_context_artifact)
    obj = _object_by_id(bundle, decision_object_id)
    existing, path = load_stage4_context_lens_checkpoint(
        project_root=project_root,
        stage3_context_artifact=stage3_context_artifact,
        decision_object_id=decision_object_id,
        lens_type=lens_type,
    )
    if existing is None:
        raise Stage4ContextArtifactError(
            "Semantic repair requires the exact prior contextual Stage 4 checkpoint"
        )
    if existing.get("assessment_sha256") != previous_assessment_sha256:
        raise Stage4ContextArtifactError(
            "Semantic repair prior-assessment hash mismatch; refusing checkpoint replacement"
        )

    parsed = validate_context_lens_assessment(
        assessment,
        decision_object=obj,
        expected_lens_type=lens_type,
    )
    output = parsed.model_dump(mode="json")
    checkpoint: dict[str, Any] = {
        "schema_version": STAGE4_CONTEXT_LENS_CHECKPOINT_SCHEMA_VERSION,
        "case_id": identity["case_id"],
        "scenario_id": obj.scenario_id,
        "input_fingerprint": identity["input_fingerprint"],
        "decision_object_id": decision_object_id,
        "lens_type": lens_type.value,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "assessment_sha256": _sha256_json(output),
        "assessment": output,
        "semantic_repair": {
            "count": 1,
            "reason": repair_reason,
            "replaced_assessment_sha256": previous_assessment_sha256,
        },
    }
    checkpoint["checkpoint_sha256"] = _sha256_json(checkpoint)

    tmp = path.with_suffix(path.suffix + ".repair.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(checkpoint, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    tmp.replace(path)
    return path


def clear_stage4_context_lens_checkpoints(
    *,
    project_root: str | Path,
    stage3_context_artifact: dict[str, Any],
) -> None:
    identity = build_stage4_context_identity(stage3_context_artifact=stage3_context_artifact)
    bundle = _validate_stage3_context_artifact(stage3_context_artifact)
    for obj in bundle.contextual_decision_objects:
        for lens_type in LensType:
            path = stage4_context_checkpoint_path(
                project_root=project_root,
                identity=identity,
                decision_object_id=obj.decision_object_id,
                lens_type=lens_type,
            )
            if path.exists():
                path.unlink()
            try:
                path.parent.rmdir()
            except OSError:
                pass


def load_frozen_stage4_context_artifact(
    *,
    project_root: str | Path,
    stage3_context_artifact: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any], Path]:
    identity = build_stage4_context_identity(stage3_context_artifact=stage3_context_artifact)
    path = stage4_context_artifact_path(project_root=project_root, identity=identity)
    if not path.exists():
        return None, identity, path
    artifact = json.loads(path.read_text(encoding="utf-8"))
    stored = artifact.get("artifact_sha256")
    unhashed = {key: value for key, value in artifact.items() if key != "artifact_sha256"}
    if not stored or stored != _sha256_json(unhashed):
        raise Stage4ContextArtifactError("Frozen contextual Stage 4 artifact hash mismatch")
    for key, expected in identity.items():
        if artifact.get(key) != expected:
            raise Stage4ContextArtifactError(f"Frozen contextual Stage 4 identity mismatch for {key}")
    stage3_bundle = _validate_stage3_context_artifact(stage3_context_artifact)
    bundle = validate_context_stage4_bundle(
        artifact.get("evaluation_bundle"),
        decision_objects=stage3_bundle.contextual_decision_objects,
    )
    if artifact.get("stage4_output_sha256") != _sha256_json(bundle.model_dump(mode="json")):
        raise Stage4ContextArtifactError("Frozen contextual Stage 4 output hash mismatch")
    return artifact, identity, path


def freeze_stage4_context_artifact(
    *,
    project_root: str | Path,
    stage3_context_artifact: dict[str, Any],
    evaluation_bundle: Stage4EvaluationBundle | dict[str, Any],
) -> tuple[dict[str, Any], Path]:
    identity = build_stage4_context_identity(stage3_context_artifact=stage3_context_artifact)
    stage3_bundle = _validate_stage3_context_artifact(stage3_context_artifact)
    bundle = validate_context_stage4_bundle(
        evaluation_bundle,
        decision_objects=stage3_bundle.contextual_decision_objects,
    )
    output = bundle.model_dump(mode="json")
    artifact: dict[str, Any] = {
        **identity,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage4_output_sha256": _sha256_json(output),
        "evaluation_bundle": output,
    }
    artifact["artifact_sha256"] = _sha256_json(artifact)
    path = stage4_context_artifact_path(project_root=project_root, identity=identity)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(artifact, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError:
        existing, _, _ = load_frozen_stage4_context_artifact(
            project_root=project_root,
            stage3_context_artifact=stage3_context_artifact,
        )
        if existing is None or existing.get("stage4_output_sha256") != artifact["stage4_output_sha256"]:
            raise Stage4ContextArtifactError("Conflicting contextual Stage 4 artifact for exact identity")
        return existing, path
    return artifact, path

