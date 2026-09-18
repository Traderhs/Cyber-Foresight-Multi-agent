from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from Stage1.runtime import get_experiment_runtime_profile, get_legacy_one_slot_runtime_profile
from Stage1.schema import CriticType, build_constrained_critic_response_schema
from Stage2.prompts import (
    STAGE2_ATTACK_ROLE_CONTEXT,
    STAGE2_DEBATE_APPENDIX,
    STAGE2_DEBATE_USER_PROMPT,
    STAGE2_DEFENSE_ROLE_CONTEXT,
    STAGE2_MEDIATOR_SYSTEM_PROMPT,
    STAGE2_MEDIATOR_USER_PROMPT,
    STAGE2_POST_ASSESSMENT_APPENDIX,
    STAGE2_POST_USER_PROMPT,
)
from Stage2.schema import (
    STAGE2_MAX_ROUNDS,
    STAGE2_SEMANTIC_VALIDATION_VERSION,
    CriticDebateTurn,
    MediatorSummary,
    Stage2DebateResult,
    build_constrained_debate_turn_schema,
    build_constrained_mediator_schema,
    validate_stage2_result,
)


STAGE2_ARTIFACT_SCHEMA_VERSION = "stage2-artifact-v9"
STAGE2_STEP_CHECKPOINT_SCHEMA_VERSION = "stage2-step-checkpoint-v2"
STAGE2_FOLLOWUP_RULE_VERSION = "mediator-adjudication-routes-round2-v1"
STAGE2_ROUND2_EVIDENCE_RULE_VERSION = "round2-surfaced-evidence-only-v1"
STAGE2_PROMPT_VISIBILITY_RULE_VERSION = "mediator-round2-post-surfaced-evidence-view-v2"
STAGE2_EXECUTION_ORDER_VERSION = "parallel-attack-defense-mediator-parallel-post-v2"
STAGE2_LEGACY_EXECUTION_ORDER_VERSION = "attack-defense-mediator-post-attack-defense-v1"


class Stage2ArtifactError(ValueError):
    """Raised when a Stage 2 artifact/checkpoint violates its exact contract."""


def _assert_pre_assessments_match_stage1_artifact(
    *,
    stage1_artifact: dict[str, Any],
    attack_pre_assessment: dict[str, Any],
    defense_pre_assessment: dict[str, Any],
) -> None:
    assessments = stage1_artifact.get("assessments") or {}
    if attack_pre_assessment != assessments.get("attack_feasibility"):
        raise Stage2ArtifactError(
            "Stage 2 Attack pre-assessment does not match the bound frozen Stage 1 artifact"
        )
    if defense_pre_assessment != assessments.get("defense_robustness"):
        raise Stage2ArtifactError(
            "Stage 2 Defense pre-assessment does not match the bound frozen Stage 1 artifact"
        )


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_text(_canonical_json(value))


def _safe_case_id(case_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", case_id).strip("._")
    return safe or "case"


def build_stage2_identity(
    *,
    evidence_pack: dict[str, Any],
    forecast_data: str,
    stage1_artifact: dict[str, Any],
    runtime_profile: dict[str, Any] | None = None,
    execution_order_version: str | None = None,
) -> dict[str, Any]:
    case_id = str(evidence_pack.get("case_id") or "")
    if not case_id:
        raise Stage2ArtifactError("Stage 0 EvidencePack is missing case_id")
    stage1_hash = str(stage1_artifact.get("artifact_sha256") or "")
    stage1_fingerprint = str(stage1_artifact.get("input_fingerprint") or "")
    if not stage1_hash or not stage1_fingerprint:
        raise Stage2ArtifactError("Stage 2 requires a validated frozen Stage 1 artifact")
    assessments = stage1_artifact.get("assessments") or {}
    attack_pre = assessments.get("attack_feasibility")
    defense_pre = assessments.get("defense_robustness")
    if not attack_pre or not defense_pre:
        raise Stage2ArtifactError("Frozen Stage 1 artifact is missing one or both critic assessments")

    identity: dict[str, Any] = {
        "schema_version": STAGE2_ARTIFACT_SCHEMA_VERSION,
        "semantic_validation_version": STAGE2_SEMANTIC_VALIDATION_VERSION,
        "case_id": case_id,
        "stage1_artifact_sha256": stage1_hash,
        "stage1_input_fingerprint": stage1_fingerprint,
        "stage0_evidence_pack_sha256": _sha256_json(evidence_pack),
        "forecast_data_sha256": _sha256_text(forecast_data),
        "max_rounds": STAGE2_MAX_ROUNDS,
        "followup_rule_version": STAGE2_FOLLOWUP_RULE_VERSION,
        "round2_evidence_rule_version": STAGE2_ROUND2_EVIDENCE_RULE_VERSION,
        "prompt_visibility_rule_version": STAGE2_PROMPT_VISIBILITY_RULE_VERSION,
        "execution_order_version": execution_order_version or STAGE2_EXECUTION_ORDER_VERSION,
        "prompt_hashes": {
            "stage2_attack_role_context_sha256": _sha256_text(STAGE2_ATTACK_ROLE_CONTEXT),
            "stage2_defense_role_context_sha256": _sha256_text(STAGE2_DEFENSE_ROLE_CONTEXT),
            "debate_appendix_sha256": _sha256_text(STAGE2_DEBATE_APPENDIX),
            "debate_user_prompt_sha256": _sha256_text(STAGE2_DEBATE_USER_PROMPT),
            "mediator_system_prompt_sha256": _sha256_text(STAGE2_MEDIATOR_SYSTEM_PROMPT),
            "mediator_user_prompt_sha256": _sha256_text(STAGE2_MEDIATOR_USER_PROMPT),
            "post_appendix_sha256": _sha256_text(STAGE2_POST_ASSESSMENT_APPENDIX),
            "post_user_prompt_sha256": _sha256_text(STAGE2_POST_USER_PROMPT),
        },
        "output_schema_hashes": {
            "critic_debate_turn_sha256": _sha256_json(CriticDebateTurn.model_json_schema()),
            "mediator_summary_sha256": _sha256_json(MediatorSummary.model_json_schema()),
            "stage2_result_sha256": _sha256_json(Stage2DebateResult.model_json_schema()),
        },
        "constrained_output_schema_hashes": {
            "round1_attack_sha256": _sha256_json(
                build_constrained_debate_turn_schema(
                    evidence_pack=evidence_pack,
                    opponent_assessment=defense_pre,
                    expected_responder=CriticType.ATTACK_FEASIBILITY,
                    round_index=1,
                )
            ),
            "round1_defense_sha256": _sha256_json(
                build_constrained_debate_turn_schema(
                    evidence_pack=evidence_pack,
                    opponent_assessment=attack_pre,
                    expected_responder=CriticType.DEFENSE_ROBUSTNESS,
                    round_index=1,
                )
            ),
            # Round 2 is narrowed at runtime to evidence surfaced in the frozen
            # pre-assessments + round 1. The exact dynamic schema is hashed in
            # that step checkpoint's dependency payload.
            "round2_attack_base_sha256": _sha256_json(
                build_constrained_debate_turn_schema(
                    evidence_pack=evidence_pack,
                    opponent_assessment=defense_pre,
                    expected_responder=CriticType.ATTACK_FEASIBILITY,
                    round_index=2,
                )
            ),
            "round2_defense_base_sha256": _sha256_json(
                build_constrained_debate_turn_schema(
                    evidence_pack=evidence_pack,
                    opponent_assessment=attack_pre,
                    expected_responder=CriticType.DEFENSE_ROBUSTNESS,
                    round_index=2,
                )
            ),
            # The Mediator's actual runtime schema is narrowed again from this
            # base contract using evidence IDs surfaced by generated critic
            # turns. That exact dynamic schema is hashed inside the step's
            # dependency payload/checkpoint rather than pretending it is known
            # before generation starts.
            "round1_mediator_base_sha256": _sha256_json(
                build_constrained_mediator_schema(
                    evidence_pack=evidence_pack,
                    attack_assessment=attack_pre,
                    defense_assessment=defense_pre,
                    round_index=1,
                )
            ),
            "round2_mediator_base_sha256": _sha256_json(
                build_constrained_mediator_schema(
                    evidence_pack=evidence_pack,
                    attack_assessment=attack_pre,
                    defense_assessment=defense_pre,
                    round_index=2,
                )
            ),
            # Post-assessment is likewise narrowed at runtime to evidence that
            # was actually surfaced before/during deliberation. These are base
            # Stage 1-compatible schemas; exact dynamic schemas are checkpointed.
            "post_attack_base_sha256": _sha256_json(
                build_constrained_critic_response_schema(
                    evidence_pack=evidence_pack,
                    expected_critic_type=CriticType.ATTACK_FEASIBILITY,
                )
            ),
            "post_defense_base_sha256": _sha256_json(
                build_constrained_critic_response_schema(
                    evidence_pack=evidence_pack,
                    expected_critic_type=CriticType.DEFENSE_ROBUSTNESS,
                )
            ),
        },
        # Stage 2 deliberately reuses the exact Stage 1 model/inference profile.
        "runtime": runtime_profile or get_experiment_runtime_profile(),
    }
    identity["input_fingerprint"] = _sha256_json(identity)
    return identity


def stage2_artifact_path(*, project_root: str | Path, identity: dict[str, Any]) -> Path:
    root = Path(project_root).resolve()
    return (
        root
        / "Multi-Agent"
        / "Results"
        / "Stage2"
        / "artifacts"
        / _safe_case_id(str(identity["case_id"]))
        / f"{identity['input_fingerprint']}.json"
    )


def stage2_step_checkpoint_path(
    *,
    project_root: str | Path,
    identity: dict[str, Any],
    step_name: str,
) -> Path:
    safe_step = re.sub(r"[^A-Za-z0-9._-]+", "_", step_name).strip("._")
    if not safe_step:
        raise Stage2ArtifactError("Stage 2 checkpoint step_name is empty")
    root = Path(project_root).resolve()
    return (
        root
        / "Multi-Agent"
        / "Results"
        / "Stage2"
        / "checkpoints"
        / _safe_case_id(str(identity["case_id"]))
        / str(identity["input_fingerprint"])
        / f"{safe_step}.json"
    )


def load_stage2_step_checkpoint(
    *,
    project_root: str | Path,
    identity: dict[str, Any],
    step_name: str,
    dependency_payload: Any,
) -> tuple[dict[str, Any] | None, Path]:
    path = stage2_step_checkpoint_path(
        project_root=project_root,
        identity=identity,
        step_name=step_name,
    )
    if not path.exists():
        return None, path
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    stored_hash = checkpoint.get("checkpoint_sha256")
    unhashed = {key: value for key, value in checkpoint.items() if key != "checkpoint_sha256"}
    if not stored_hash or stored_hash != _sha256_json(unhashed):
        raise Stage2ArtifactError(f"Stage 2 checkpoint hash mismatch: {path}")
    if checkpoint.get("schema_version") != STAGE2_STEP_CHECKPOINT_SCHEMA_VERSION:
        raise Stage2ArtifactError("Stage 2 checkpoint schema version mismatch")
    if checkpoint.get("case_id") != identity["case_id"]:
        raise Stage2ArtifactError("Stage 2 checkpoint case_id mismatch")
    if checkpoint.get("input_fingerprint") != identity["input_fingerprint"]:
        raise Stage2ArtifactError("Stage 2 checkpoint fingerprint mismatch")
    if checkpoint.get("step_name") != step_name:
        raise Stage2ArtifactError("Stage 2 checkpoint step_name mismatch")
    if checkpoint.get("payload_sha256") != _sha256_json(checkpoint.get("payload")):
        raise Stage2ArtifactError("Stage 2 checkpoint payload hash mismatch")
    expected_dependency_sha256 = _sha256_json(dependency_payload)
    if checkpoint.get("dependency_sha256") != expected_dependency_sha256:
        # This is a valid checkpoint for the same experiment identity, but it
        # was produced from different upstream generated content. Partial
        # checkpoints are resumability aids rather than immutable final
        # research artifacts, so invalidate the stale downstream step and let
        # it regenerate under the current exact dependency. Integrity/hash
        # failures above remain hard errors and are never silently discarded.
        path.unlink(missing_ok=True)
        return None, path
    return checkpoint, path


def freeze_stage2_step_checkpoint(
    *,
    project_root: str | Path,
    identity: dict[str, Any],
    step_name: str,
    payload: dict[str, Any],
    dependency_payload: Any,
) -> Path:
    checkpoint: dict[str, Any] = {
        "schema_version": STAGE2_STEP_CHECKPOINT_SCHEMA_VERSION,
        "case_id": identity["case_id"],
        "input_fingerprint": identity["input_fingerprint"],
        "step_name": step_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dependency_sha256": _sha256_json(dependency_payload),
        "payload_sha256": _sha256_json(payload),
        "payload": payload,
    }
    checkpoint["checkpoint_sha256"] = _sha256_json(checkpoint)
    path = stage2_step_checkpoint_path(
        project_root=project_root,
        identity=identity,
        step_name=step_name,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(checkpoint, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError:
        existing, _ = load_stage2_step_checkpoint(
            project_root=project_root,
            identity=identity,
            step_name=step_name,
            dependency_payload=dependency_payload,
        )
        if existing is None:
            # load_stage2_step_checkpoint removes a valid-but-stale checkpoint
            # whose upstream dependency no longer matches. Retry the exclusive
            # creation once with the current payload.
            with path.open("x", encoding="utf-8", newline="\n") as handle:
                json.dump(checkpoint, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
        elif existing.get("payload_sha256") != checkpoint["payload_sha256"]:
            raise Stage2ArtifactError(
                "A Stage 2 step checkpoint already exists for this exact identity with a different payload"
            )
    return path


def clear_stage2_step_checkpoints(*, project_root: str | Path, identity: dict[str, Any]) -> None:
    root = (
        Path(project_root).resolve()
        / "Multi-Agent"
        / "Results"
        / "Stage2"
        / "checkpoints"
        / _safe_case_id(str(identity["case_id"]))
        / str(identity["input_fingerprint"])
    )
    if not root.exists():
        return
    for path in root.glob("*.json"):
        path.unlink()
    try:
        root.rmdir()
    except OSError:
        pass


def load_frozen_stage2_artifact(
    *,
    project_root: str | Path,
    evidence_pack: dict[str, Any],
    forecast_data: str,
    stage1_artifact: dict[str, Any],
    attack_pre_assessment: dict[str, Any],
    defense_pre_assessment: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any], Path]:
    _assert_pre_assessments_match_stage1_artifact(
        stage1_artifact=stage1_artifact,
        attack_pre_assessment=attack_pre_assessment,
        defense_pre_assessment=defense_pre_assessment,
    )
    identity = build_stage2_identity(
        evidence_pack=evidence_pack,
        forecast_data=forecast_data,
        stage1_artifact=stage1_artifact,
    )
    path = stage2_artifact_path(project_root=project_root, identity=identity)
    if not path.exists():
        legacy_identity = build_stage2_identity(
            evidence_pack=evidence_pack,
            forecast_data=forecast_data,
            stage1_artifact=stage1_artifact,
            runtime_profile=get_legacy_one_slot_runtime_profile(),
            execution_order_version=STAGE2_LEGACY_EXECUTION_ORDER_VERSION,
        )
        legacy_path = stage2_artifact_path(project_root=project_root, identity=legacy_identity)
        if not legacy_path.exists():
            return None, identity, path
        identity = legacy_identity
        path = legacy_path
    artifact = json.loads(path.read_text(encoding="utf-8"))
    stored_hash = artifact.get("artifact_sha256")
    unhashed = {key: value for key, value in artifact.items() if key != "artifact_sha256"}
    if not stored_hash or stored_hash != _sha256_json(unhashed):
        raise Stage2ArtifactError("Frozen Stage 2 artifact hash mismatch")
    for key, expected in identity.items():
        if artifact.get(key) != expected:
            raise Stage2ArtifactError(f"Frozen Stage 2 artifact identity mismatch for {key}")
    parsed = validate_stage2_result(
        artifact.get("debate_result"),
        evidence_pack=evidence_pack,
        attack_pre_assessment=attack_pre_assessment,
        defense_pre_assessment=defense_pre_assessment,
    )
    if artifact.get("stage2_output_sha256") != _sha256_json(parsed.model_dump(mode="json")):
        raise Stage2ArtifactError("Frozen Stage 2 output hash mismatch")
    return artifact, identity, path


def freeze_stage2_artifact(
    *,
    project_root: str | Path,
    evidence_pack: dict[str, Any],
    forecast_data: str,
    stage1_artifact: dict[str, Any],
    attack_pre_assessment: dict[str, Any],
    defense_pre_assessment: dict[str, Any],
    debate_result: Stage2DebateResult | dict[str, Any],
) -> tuple[dict[str, Any], Path]:
    _assert_pre_assessments_match_stage1_artifact(
        stage1_artifact=stage1_artifact,
        attack_pre_assessment=attack_pre_assessment,
        defense_pre_assessment=defense_pre_assessment,
    )
    identity = build_stage2_identity(
        evidence_pack=evidence_pack,
        forecast_data=forecast_data,
        stage1_artifact=stage1_artifact,
    )
    parsed = validate_stage2_result(
        debate_result,
        evidence_pack=evidence_pack,
        attack_pre_assessment=attack_pre_assessment,
        defense_pre_assessment=defense_pre_assessment,
    )
    output = parsed.model_dump(mode="json")
    artifact: dict[str, Any] = {
        **identity,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_snapshot_id": evidence_pack.get("source_snapshot_id"),
        "stage2_output_sha256": _sha256_json(output),
        "debate_result": output,
    }
    artifact["artifact_sha256"] = _sha256_json(artifact)
    path = stage2_artifact_path(project_root=project_root, identity=identity)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(artifact, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError:
        existing, _, _ = load_frozen_stage2_artifact(
            project_root=project_root,
            evidence_pack=evidence_pack,
            forecast_data=forecast_data,
            stage1_artifact=stage1_artifact,
            attack_pre_assessment=attack_pre_assessment,
            defense_pre_assessment=defense_pre_assessment,
        )
        if existing is None or existing.get("stage2_output_sha256") != artifact["stage2_output_sha256"]:
            raise Stage2ArtifactError(
                "A Stage 2 artifact already exists for this exact experiment identity with a different output"
            )
        artifact = existing
    return artifact, path
