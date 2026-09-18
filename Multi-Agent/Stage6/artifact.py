from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from Stage6.builder import build_stage6_bundle, build_stage6_frames
from Stage6.prompts import STAGE6_SYNTHESIS_SYSTEM_PROMPT, STAGE6_SYNTHESIS_USER_PROMPT
from Stage6.runtime import get_stage6_runtime_profile
from Stage6.schema import (
    STAGE6_DECISION_POLICY_VERSION,
    STAGE6_SEMANTIC_VALIDATION_VERSION,
    STAGE6_SYNTHESIS_CONTRACT_VERSION,
    DecisionSynthesis,
    Stage6SynthesisBundle,
    Stage6ValidationError,
    SynthesisNarrative,
    build_constrained_synthesis_narrative_schema,
    validate_synthesis_narrative,
)


STAGE6_ARTIFACT_SCHEMA_VERSION = "stage6-artifact-v5"
STAGE6_CHECKPOINT_SCHEMA_VERSION = "stage6-synthesis-checkpoint-v5"
STAGE6_GENERATION_CONTRACT_VERSION = "policy-conditional-evidence-grounded-strategic-report-v5"
_VALIDATOR_ONLY_REUSE_COMPATIBLE_VERSIONS = ("stage6-semantic-validation-v11",)


class Stage6ArtifactError(ValueError):
    """Raised when a Stage 6 artifact/checkpoint violates exact provenance."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_text(_canonical_json(value))


def _safe(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return safe or "item"


def _require_artifact_hash(artifact: dict[str, Any], *, label: str) -> str:
    stored = str(artifact.get("artifact_sha256") or "")
    if not stored:
        raise Stage6ArtifactError(f"Stage 6 requires {label} artifact_sha256")
    unhashed = {key: value for key, value in artifact.items() if key != "artifact_sha256"}
    if stored != _sha256_json(unhashed):
        raise Stage6ArtifactError(f"Stage 6 rejected {label} artifact content hash mismatch")
    return stored


def _upstream_bundles(
    *,
    stage3_context_artifact: dict[str, Any],
    stage4_context_artifact: dict[str, Any],
    stage5_artifact: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    stage3 = stage3_context_artifact.get("contextual_decision_bundle")
    stage4 = stage4_context_artifact.get("evaluation_bundle")
    stage5 = stage5_artifact.get("diagnostic_bundle")
    if stage3 is None or stage4 is None or stage5 is None:
        raise Stage6ArtifactError("Stage 6 requires complete Stage 3/4/5 upstream bundles")
    return stage3, stage4, stage5


def build_stage6_identity(
    *,
    stage3_context_artifact: dict[str, Any],
    stage4_context_artifact: dict[str, Any],
    stage5_artifact: dict[str, Any],
    semantic_validation_version: str | None = None,
) -> dict[str, Any]:
    stage3_hash = _require_artifact_hash(stage3_context_artifact, label="contextual Stage 3")
    stage4_hash = _require_artifact_hash(stage4_context_artifact, label="contextual Stage 4")
    stage5_hash = _require_artifact_hash(stage5_artifact, label="Stage 5")
    stage3, stage4, stage5 = _upstream_bundles(
        stage3_context_artifact=stage3_context_artifact,
        stage4_context_artifact=stage4_context_artifact,
        stage5_artifact=stage5_artifact,
    )
    frames = build_stage6_frames(
        stage3_context_bundle=stage3,
        stage4_context_bundle=stage4,
        stage5_diagnostic_bundle=stage5,
    )
    case_ids = {
        str(stage3_context_artifact.get("case_id") or ""),
        str(stage4_context_artifact.get("case_id") or ""),
        str(stage5_artifact.get("case_id") or ""),
    }
    if len(case_ids) != 1 or not next(iter(case_ids)):
        raise Stage6ArtifactError("Stage 6 upstream artifact case_id mismatch")
    case_id = next(iter(case_ids))
    frame_hashes = {
        frame["deterministic_output"]["decision_object_id"]: _sha256_json(frame)
        for frame in frames
    }
    constrained_schema_hashes = {
        frame["deterministic_output"]["decision_object_id"]: _sha256_json(
            build_constrained_synthesis_narrative_schema(frame=frame)
        )
        for frame in frames
    }
    identity: dict[str, Any] = {
        "schema_version": STAGE6_ARTIFACT_SCHEMA_VERSION,
        "semantic_validation_version": (
            semantic_validation_version or STAGE6_SEMANTIC_VALIDATION_VERSION
        ),
        "synthesis_contract_version": STAGE6_SYNTHESIS_CONTRACT_VERSION,
        "decision_policy_version": STAGE6_DECISION_POLICY_VERSION,
        "generation_contract_version": STAGE6_GENERATION_CONTRACT_VERSION,
        "case_id": case_id,
        "stage3_context_artifact_sha256": stage3_hash,
        "stage3_context_output_sha256": stage3_context_artifact.get("stage3_contextual_output_sha256"),
        "stage4_context_artifact_sha256": stage4_hash,
        "stage4_context_output_sha256": stage4_context_artifact.get("stage4_output_sha256"),
        "stage5_artifact_sha256": stage5_hash,
        "stage5_output_sha256": stage5_artifact.get("stage5_output_sha256"),
        "frame_sha256": frame_hashes,
        "prompt_hashes": {
            "system_prompt_sha256": _sha256_text(STAGE6_SYNTHESIS_SYSTEM_PROMPT),
            "user_prompt_sha256": _sha256_text(STAGE6_SYNTHESIS_USER_PROMPT),
        },
        "narrative_schema_sha256": _sha256_json(SynthesisNarrative.model_json_schema()),
        "constrained_narrative_schema_sha256": constrained_schema_hashes,
        "output_schema_sha256": _sha256_json(Stage6SynthesisBundle.model_json_schema()),
        "runtime": get_stage6_runtime_profile(),
    }
    for key in (
        "stage3_context_output_sha256",
        "stage4_context_output_sha256",
        "stage5_output_sha256",
    ):
        if not identity[key]:
            raise Stage6ArtifactError(f"Stage 6 requires {key}")
    identity["input_fingerprint"] = _sha256_json(identity)
    return identity


def _legacy_validator_only_identities(
    *,
    stage3_context_artifact: dict[str, Any],
    stage4_context_artifact: dict[str, Any],
    stage5_artifact: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return prior Stage 6 identities eligible for validation-only reuse.

    v11 -> v12 changes only how already-written inline evidence citation
    brackets are parsed.  It does not change the prompt, model/runtime,
    deterministic policy, narrative schema, or upstream frame.  Exact v11
    outputs can therefore be reused only after they pass the current validator.
    Any other identity change remains incompatible and requires regeneration.
    """

    return [
        build_stage6_identity(
            stage3_context_artifact=stage3_context_artifact,
            stage4_context_artifact=stage4_context_artifact,
            stage5_artifact=stage5_artifact,
            semantic_validation_version=version,
        )
        for version in _VALIDATOR_ONLY_REUSE_COMPATIBLE_VERSIONS
    ]


def stage6_artifact_path(*, project_root: str | Path, identity: dict[str, Any]) -> Path:
    return (
        Path(project_root).resolve()
        / "Multi-Agent"
        / "Results"
        / "Stage6"
        / "artifacts"
        / _safe(str(identity["case_id"]))
        / f"{identity['input_fingerprint']}.json"
    )


def stage6_checkpoint_path(
    *,
    project_root: str | Path,
    identity: dict[str, Any],
    decision_object_id: str,
) -> Path:
    return (
        Path(project_root).resolve()
        / "Multi-Agent"
        / "Results"
        / "Stage6"
        / "checkpoints"
        / _safe(str(identity["case_id"]))
        / str(identity["input_fingerprint"])
        / f"{_safe(decision_object_id)}.json"
    )


def _frames(
    *,
    stage3_context_artifact: dict[str, Any],
    stage4_context_artifact: dict[str, Any],
    stage5_artifact: dict[str, Any],
) -> list[dict[str, Any]]:
    stage3, stage4, stage5 = _upstream_bundles(
        stage3_context_artifact=stage3_context_artifact,
        stage4_context_artifact=stage4_context_artifact,
        stage5_artifact=stage5_artifact,
    )
    return build_stage6_frames(
        stage3_context_bundle=stage3,
        stage4_context_bundle=stage4,
        stage5_diagnostic_bundle=stage5,
    )


def _frame_by_id(frames: list[dict[str, Any]], decision_object_id: str) -> dict[str, Any]:
    matches = [
        frame
        for frame in frames
        if frame["deterministic_output"]["decision_object_id"] == decision_object_id
    ]
    if len(matches) != 1:
        raise Stage6ArtifactError(f"Unknown Stage 6 decision_object_id: {decision_object_id!r}")
    return matches[0]


def load_stage6_checkpoint(
    *,
    project_root: str | Path,
    stage3_context_artifact: dict[str, Any],
    stage4_context_artifact: dict[str, Any],
    stage5_artifact: dict[str, Any],
    decision_object_id: str,
) -> tuple[dict[str, Any] | None, Path]:
    identity = build_stage6_identity(
        stage3_context_artifact=stage3_context_artifact,
        stage4_context_artifact=stage4_context_artifact,
        stage5_artifact=stage5_artifact,
    )
    frames = _frames(
        stage3_context_artifact=stage3_context_artifact,
        stage4_context_artifact=stage4_context_artifact,
        stage5_artifact=stage5_artifact,
    )
    frame = _frame_by_id(frames, decision_object_id)
    path = stage6_checkpoint_path(
        project_root=project_root,
        identity=identity,
        decision_object_id=decision_object_id,
    )
    if not path.exists():
        for legacy_identity in _legacy_validator_only_identities(
            stage3_context_artifact=stage3_context_artifact,
            stage4_context_artifact=stage4_context_artifact,
            stage5_artifact=stage5_artifact,
        ):
            legacy_path = stage6_checkpoint_path(
                project_root=project_root,
                identity=legacy_identity,
                decision_object_id=decision_object_id,
            )
            if not legacy_path.exists():
                continue
            legacy_checkpoint = json.loads(legacy_path.read_text(encoding="utf-8"))
            stored = legacy_checkpoint.get("checkpoint_sha256")
            unhashed = {
                key: value
                for key, value in legacy_checkpoint.items()
                if key != "checkpoint_sha256"
            }
            if not stored or stored != _sha256_json(unhashed):
                raise Stage6ArtifactError("Legacy Stage 6 checkpoint hash mismatch")
            if legacy_checkpoint.get("schema_version") != STAGE6_CHECKPOINT_SCHEMA_VERSION:
                raise Stage6ArtifactError("Legacy Stage 6 checkpoint schema mismatch")
            if legacy_checkpoint.get("input_fingerprint") != legacy_identity["input_fingerprint"]:
                raise Stage6ArtifactError("Legacy Stage 6 checkpoint fingerprint mismatch")
            if legacy_checkpoint.get("decision_object_id") != decision_object_id:
                raise Stage6ArtifactError("Legacy Stage 6 checkpoint decision_object_id mismatch")
            try:
                narrative = validate_synthesis_narrative(
                    legacy_checkpoint.get("narrative"), frame=frame
                )
            except Stage6ValidationError:
                # Validator-only reuse is opportunistic. An old generated
                # narrative that does not satisfy the current semantic contract
                # is an incompatible cache candidate, not a batch-wide failure.
                continue
            if legacy_checkpoint.get("narrative_sha256") != _sha256_json(
                narrative.model_dump(mode="json")
            ):
                raise Stage6ArtifactError("Legacy Stage 6 checkpoint narrative hash mismatch")
            freeze_stage6_checkpoint(
                project_root=project_root,
                stage3_context_artifact=stage3_context_artifact,
                stage4_context_artifact=stage4_context_artifact,
                stage5_artifact=stage5_artifact,
                decision_object_id=decision_object_id,
                narrative=narrative,
            )
            return load_stage6_checkpoint(
                project_root=project_root,
                stage3_context_artifact=stage3_context_artifact,
                stage4_context_artifact=stage4_context_artifact,
                stage5_artifact=stage5_artifact,
                decision_object_id=decision_object_id,
            )
        return None, path
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    stored = checkpoint.get("checkpoint_sha256")
    unhashed = {key: value for key, value in checkpoint.items() if key != "checkpoint_sha256"}
    if not stored or stored != _sha256_json(unhashed):
        raise Stage6ArtifactError("Stage 6 checkpoint hash mismatch")
    if checkpoint.get("schema_version") != STAGE6_CHECKPOINT_SCHEMA_VERSION:
        raise Stage6ArtifactError("Stage 6 checkpoint schema mismatch")
    if checkpoint.get("input_fingerprint") != identity["input_fingerprint"]:
        raise Stage6ArtifactError("Stage 6 checkpoint fingerprint mismatch")
    if checkpoint.get("decision_object_id") != decision_object_id:
        raise Stage6ArtifactError("Stage 6 checkpoint decision_object_id mismatch")
    narrative = validate_synthesis_narrative(checkpoint.get("narrative"), frame=frame)
    if checkpoint.get("narrative_sha256") != _sha256_json(narrative.model_dump(mode="json")):
        raise Stage6ArtifactError("Stage 6 checkpoint narrative hash mismatch")
    return checkpoint, path


def freeze_stage6_checkpoint(
    *,
    project_root: str | Path,
    stage3_context_artifact: dict[str, Any],
    stage4_context_artifact: dict[str, Any],
    stage5_artifact: dict[str, Any],
    decision_object_id: str,
    narrative: SynthesisNarrative | dict[str, Any],
) -> Path:
    identity = build_stage6_identity(
        stage3_context_artifact=stage3_context_artifact,
        stage4_context_artifact=stage4_context_artifact,
        stage5_artifact=stage5_artifact,
    )
    frames = _frames(
        stage3_context_artifact=stage3_context_artifact,
        stage4_context_artifact=stage4_context_artifact,
        stage5_artifact=stage5_artifact,
    )
    frame = _frame_by_id(frames, decision_object_id)
    parsed = validate_synthesis_narrative(narrative, frame=frame)
    output = parsed.model_dump(mode="json")
    checkpoint: dict[str, Any] = {
        "schema_version": STAGE6_CHECKPOINT_SCHEMA_VERSION,
        "case_id": identity["case_id"],
        "scenario_id": frame["deterministic_output"]["scenario_id"],
        "input_fingerprint": identity["input_fingerprint"],
        "decision_object_id": decision_object_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "narrative_sha256": _sha256_json(output),
        "narrative": output,
    }
    checkpoint["checkpoint_sha256"] = _sha256_json(checkpoint)
    path = stage6_checkpoint_path(
        project_root=project_root,
        identity=identity,
        decision_object_id=decision_object_id,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(checkpoint, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError:
        existing, _ = load_stage6_checkpoint(
            project_root=project_root,
            stage3_context_artifact=stage3_context_artifact,
            stage4_context_artifact=stage4_context_artifact,
            stage5_artifact=stage5_artifact,
            decision_object_id=decision_object_id,
        )
        if existing is None or existing.get("narrative_sha256") != checkpoint["narrative_sha256"]:
            raise Stage6ArtifactError("Conflicting Stage 6 checkpoint for exact identity")
    return path


def _narratives_from_bundle(
    *,
    bundle: Stage6SynthesisBundle,
    frames: list[dict[str, Any]],
) -> list[SynthesisNarrative]:
    synthesis_by_id: dict[str, DecisionSynthesis] = {
        item.decision_object_id: item for item in bundle.syntheses
    }
    narratives: list[SynthesisNarrative] = []
    for frame in frames:
        frozen = frame["deterministic_output"]
        synthesis = synthesis_by_id.get(frozen["decision_object_id"])
        if synthesis is None:
            raise Stage6ArtifactError("Stage 6 bundle is missing a frozen DecisionObject synthesis")
        narrative = SynthesisNarrative(
            decision_object_id=frozen["decision_object_id"],
            decision_recommendation=frozen["decision_recommendation"],
            strategic_intelligence_report=synthesis.strategic_intelligence_report,
        )
        narratives.append(validate_synthesis_narrative(narrative, frame=frame))
    return narratives


def _promote_valid_legacy_bundle_narratives_to_checkpoints(
    *,
    project_root: str | Path,
    bundle: Stage6SynthesisBundle,
    frames: list[dict[str, Any]],
    stage3_context_artifact: dict[str, Any],
    stage4_context_artifact: dict[str, Any],
    stage5_artifact: dict[str, Any],
) -> tuple[list[SynthesisNarrative], list[str]]:
    """Salvage current-validator-valid scenario narratives from a legacy artifact.

    Stage 6 artifacts are case-level bundles, but generation/checkpointing is
    scenario-level.  A validator-only migration should therefore not discard
    two valid jurisdiction narratives merely because the third one fails the
    newer semantic validator.  Valid narratives are promoted to current
    per-scenario checkpoints; invalid narratives are left missing so normal
    resume regenerates only those scenarios.
    """

    synthesis_by_id: dict[str, DecisionSynthesis] = {
        item.decision_object_id: item for item in bundle.syntheses
    }
    valid: list[SynthesisNarrative] = []
    invalid_ids: list[str] = []
    for frame in frames:
        frozen = frame["deterministic_output"]
        decision_object_id = frozen["decision_object_id"]
        synthesis = synthesis_by_id.get(decision_object_id)
        if synthesis is None:
            invalid_ids.append(decision_object_id)
            continue
        narrative = SynthesisNarrative(
            decision_object_id=decision_object_id,
            decision_recommendation=frozen["decision_recommendation"],
            strategic_intelligence_report=synthesis.strategic_intelligence_report,
        )
        try:
            parsed = validate_synthesis_narrative(narrative, frame=frame)
        except Stage6ValidationError:
            invalid_ids.append(decision_object_id)
            continue
        freeze_stage6_checkpoint(
            project_root=project_root,
            stage3_context_artifact=stage3_context_artifact,
            stage4_context_artifact=stage4_context_artifact,
            stage5_artifact=stage5_artifact,
            decision_object_id=decision_object_id,
            narrative=parsed,
        )
        valid.append(parsed)
    return valid, invalid_ids


def load_frozen_stage6_artifact(
    *,
    project_root: str | Path,
    stage3_context_artifact: dict[str, Any],
    stage4_context_artifact: dict[str, Any],
    stage5_artifact: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any], Path]:
    identity = build_stage6_identity(
        stage3_context_artifact=stage3_context_artifact,
        stage4_context_artifact=stage4_context_artifact,
        stage5_artifact=stage5_artifact,
    )
    path = stage6_artifact_path(project_root=project_root, identity=identity)
    if not path.exists():
        for legacy_identity in _legacy_validator_only_identities(
            stage3_context_artifact=stage3_context_artifact,
            stage4_context_artifact=stage4_context_artifact,
            stage5_artifact=stage5_artifact,
        ):
            legacy_path = stage6_artifact_path(
                project_root=project_root,
                identity=legacy_identity,
            )
            if not legacy_path.exists():
                continue
            legacy_artifact = json.loads(legacy_path.read_text(encoding="utf-8"))
            stored = legacy_artifact.get("artifact_sha256")
            unhashed = {
                key: value
                for key, value in legacy_artifact.items()
                if key != "artifact_sha256"
            }
            if not stored or stored != _sha256_json(unhashed):
                raise Stage6ArtifactError("Legacy frozen Stage 6 artifact content hash mismatch")
            for key, expected in legacy_identity.items():
                if legacy_artifact.get(key) != expected:
                    raise Stage6ArtifactError(
                        f"Legacy frozen Stage 6 identity mismatch for {key}"
                    )
            bundle = Stage6SynthesisBundle.model_validate(
                legacy_artifact.get("synthesis_bundle")
            )
            frames = _frames(
                stage3_context_artifact=stage3_context_artifact,
                stage4_context_artifact=stage4_context_artifact,
                stage5_artifact=stage5_artifact,
            )
            try:
                narratives = _narratives_from_bundle(bundle=bundle, frames=frames)
            except Stage6ValidationError:
                valid_narratives, _ = _promote_valid_legacy_bundle_narratives_to_checkpoints(
                    project_root=project_root,
                    bundle=bundle,
                    frames=frames,
                    stage3_context_artifact=stage3_context_artifact,
                    stage4_context_artifact=stage4_context_artifact,
                    stage5_artifact=stage5_artifact,
                )
                if valid_narratives:
                    # The case-level artifact itself cannot be promoted because
                    # at least one scenario failed current validation.  The
                    # valid scenarios are now current checkpoints, so normal
                    # prepare/resume will generate only the missing scenarios.
                    return None, identity, path
                continue
            rebuilt = build_stage6_bundle(
                case_id=identity["case_id"],
                frames=frames,
                narratives=narratives,
            )
            promoted, promoted_path = freeze_stage6_artifact(
                project_root=project_root,
                stage3_context_artifact=stage3_context_artifact,
                stage4_context_artifact=stage4_context_artifact,
                stage5_artifact=stage5_artifact,
                synthesis_bundle=rebuilt,
            )
            return promoted, identity, promoted_path
        return None, identity, path
    artifact = json.loads(path.read_text(encoding="utf-8"))
    stored = artifact.get("artifact_sha256")
    unhashed = {key: value for key, value in artifact.items() if key != "artifact_sha256"}
    if not stored or stored != _sha256_json(unhashed):
        raise Stage6ArtifactError("Frozen Stage 6 artifact content hash mismatch")
    for key, expected in identity.items():
        if artifact.get(key) != expected:
            raise Stage6ArtifactError(f"Frozen Stage 6 identity mismatch for {key}")
    bundle = Stage6SynthesisBundle.model_validate(artifact.get("synthesis_bundle"))
    frames = _frames(
        stage3_context_artifact=stage3_context_artifact,
        stage4_context_artifact=stage4_context_artifact,
        stage5_artifact=stage5_artifact,
    )
    narratives = _narratives_from_bundle(bundle=bundle, frames=frames)
    rebuilt = build_stage6_bundle(case_id=identity["case_id"], frames=frames, narratives=narratives)
    output = rebuilt.model_dump(mode="json")
    if bundle.model_dump(mode="json") != output:
        raise Stage6ArtifactError("Frozen Stage 6 deterministic fields drifted from exact upstream artifacts")
    if artifact.get("stage6_output_sha256") != _sha256_json(output):
        raise Stage6ArtifactError("Frozen Stage 6 output hash mismatch")
    return artifact, identity, path


def freeze_stage6_artifact(
    *,
    project_root: str | Path,
    stage3_context_artifact: dict[str, Any],
    stage4_context_artifact: dict[str, Any],
    stage5_artifact: dict[str, Any],
    synthesis_bundle: Stage6SynthesisBundle | dict[str, Any],
) -> tuple[dict[str, Any], Path]:
    identity = build_stage6_identity(
        stage3_context_artifact=stage3_context_artifact,
        stage4_context_artifact=stage4_context_artifact,
        stage5_artifact=stage5_artifact,
    )
    bundle = (
        synthesis_bundle
        if isinstance(synthesis_bundle, Stage6SynthesisBundle)
        else Stage6SynthesisBundle.model_validate(synthesis_bundle)
    )
    frames = _frames(
        stage3_context_artifact=stage3_context_artifact,
        stage4_context_artifact=stage4_context_artifact,
        stage5_artifact=stage5_artifact,
    )
    narratives = _narratives_from_bundle(bundle=bundle, frames=frames)
    rebuilt = build_stage6_bundle(case_id=identity["case_id"], frames=frames, narratives=narratives)
    output = rebuilt.model_dump(mode="json")
    artifact: dict[str, Any] = {
        **identity,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage6_output_sha256": _sha256_json(output),
        "synthesis_bundle": output,
    }
    artifact["artifact_sha256"] = _sha256_json(artifact)
    path = stage6_artifact_path(project_root=project_root, identity=identity)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(artifact, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError:
        existing, _, _ = load_frozen_stage6_artifact(
            project_root=project_root,
            stage3_context_artifact=stage3_context_artifact,
            stage4_context_artifact=stage4_context_artifact,
            stage5_artifact=stage5_artifact,
        )
        if existing is None or existing.get("stage6_output_sha256") != artifact["stage6_output_sha256"]:
            raise Stage6ArtifactError("Conflicting frozen Stage 6 artifact for exact identity")
        return existing, path
    return artifact, path


def clear_stage6_checkpoints(
    *,
    project_root: str | Path,
    stage3_context_artifact: dict[str, Any],
    stage4_context_artifact: dict[str, Any],
    stage5_artifact: dict[str, Any],
) -> None:
    identity = build_stage6_identity(
        stage3_context_artifact=stage3_context_artifact,
        stage4_context_artifact=stage4_context_artifact,
        stage5_artifact=stage5_artifact,
    )
    for frame in _frames(
        stage3_context_artifact=stage3_context_artifact,
        stage4_context_artifact=stage4_context_artifact,
        stage5_artifact=stage5_artifact,
    ):
        path = stage6_checkpoint_path(
            project_root=project_root,
            identity=identity,
            decision_object_id=frame["deterministic_output"]["decision_object_id"],
        )
        if path.exists():
            path.unlink()
    checkpoint_root = (
        Path(project_root).resolve()
        / "Multi-Agent"
        / "Results"
        / "Stage6"
        / "checkpoints"
        / _safe(str(identity["case_id"]))
        / str(identity["input_fingerprint"])
    )
    try:
        checkpoint_root.rmdir()
    except OSError:
        pass

