from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from Stage0.context_scenarios import context_scenario_manifest
from Stage0.action_evidence import (
    ACTION_EVIDENCE_REGISTRY_SHA256,
    ACTION_EVIDENCE_REGISTRY_VERSION,
    ACTION_EVIDENCE_SNAPSHOT_ID,
)
from Stage0.decision_evidence import DECISION_EVIDENCE_RETRIEVAL_VERSION
from Stage0.decision_sources import (
    DECISION_SOURCE_REGISTRY_VERSION,
    DECISION_SOURCE_RECORDS_SHA256,
    DECISION_SOURCE_SNAPSHOT_ID,
    DECISION_SOURCE_SPEC_MANIFEST_SHA256,
)
from Stage0.main_guidance_selection import (
    MAIN_GUIDANCE_SELECTION_SHA256,
    MAIN_GUIDANCE_SELECTION_VERSION,
)
from Stage3.context_schema import (
    STAGE3_CONTEXT_SEMANTIC_VALIDATION_VERSION,
    Stage3ContextualDecisionBundle,
)
from Stage3.artifact import STAGE3_ARTIFACT_SCHEMA_VERSION
from Stage3.schema import STAGE3_SEMANTIC_VALIDATION_VERSION


STAGE3_CONTEXT_ARTIFACT_SCHEMA_VERSION = "stage3-artifact-v4"


class Stage3ContextArtifactError(ValueError):
    pass


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _safe(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return safe or "case"


def _validate_base_stage3_artifact(base_stage3_artifact: dict[str, Any], case_id: str) -> None:
    if base_stage3_artifact.get("schema_version") != STAGE3_ARTIFACT_SCHEMA_VERSION:
        raise Stage3ContextArtifactError("Contextual Stage 3 requires the current base Stage 3 artifact schema")
    if base_stage3_artifact.get("semantic_validation_version") != STAGE3_SEMANTIC_VALIDATION_VERSION:
        raise Stage3ContextArtifactError("Contextual Stage 3 requires the current base Stage 3 semantic contract")
    stored = str(base_stage3_artifact.get("artifact_sha256") or "")
    output_hash = str(base_stage3_artifact.get("stage3_output_sha256") or "")
    bundle = base_stage3_artifact.get("decision_bundle")
    if not stored or not output_hash or bundle is None:
        raise Stage3ContextArtifactError("Contextual Stage 3 requires a complete frozen base Stage 3 artifact")
    unhashed = {key: value for key, value in base_stage3_artifact.items() if key != "artifact_sha256"}
    if stored != _sha256_json(unhashed):
        raise Stage3ContextArtifactError("Base Stage 3 artifact content hash mismatch")
    if output_hash != _sha256_json(bundle):
        raise Stage3ContextArtifactError("Base Stage 3 artifact output hash mismatch")
    if str(base_stage3_artifact.get("case_id") or bundle.get("case_id") or "") != case_id:
        raise Stage3ContextArtifactError("Base Stage 3 / Stage 0 case mismatch")


def build_stage3_context_identity(
    *,
    evidence_pack: dict[str, Any],
    base_stage3_artifact: dict[str, Any],
) -> dict[str, Any]:
    case_id = str(evidence_pack.get("case_id") or "")
    if not case_id:
        raise Stage3ContextArtifactError("Contextual Stage 3 requires case_id")
    _validate_base_stage3_artifact(base_stage3_artifact, case_id)
    manifest = context_scenario_manifest()
    identity: dict[str, Any] = {
        "schema_version": STAGE3_CONTEXT_ARTIFACT_SCHEMA_VERSION,
        "semantic_validation_version": STAGE3_CONTEXT_SEMANTIC_VALIDATION_VERSION,
        "case_id": case_id,
        "source_snapshot_id": evidence_pack.get("source_snapshot_id"),
        "analysis_cutoff_date": evidence_pack.get("analysis_cutoff_date"),
        "stage0_evidence_pack_sha256": _sha256_json(evidence_pack),
        "base_stage3_artifact_sha256": base_stage3_artifact["artifact_sha256"],
        "base_stage3_output_sha256": base_stage3_artifact["stage3_output_sha256"],
        "base_stage3_input_fingerprint": base_stage3_artifact.get("input_fingerprint"),
        "scenario_set_version": manifest["scenario_set_version"],
        "scenario_manifest_sha256": manifest["manifest_sha256"],
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
        "output_schema_sha256": _sha256_json(Stage3ContextualDecisionBundle.model_json_schema()),
    }
    identity["input_fingerprint"] = _sha256_json(identity)
    return identity


def stage3_context_artifact_path(*, project_root: str | Path, identity: dict[str, Any]) -> Path:
    return (
        Path(project_root).resolve()
        / "Multi-Agent"
        / "Results"
        / "Stage3"
        / "contextual_artifacts"
        / _safe(str(identity["case_id"]))
        / f"{identity['input_fingerprint']}.json"
    )


def load_frozen_stage3_context_artifact(
    *,
    project_root: str | Path,
    evidence_pack: dict[str, Any],
    base_stage3_artifact: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any], Path]:
    identity = build_stage3_context_identity(
        evidence_pack=evidence_pack,
        base_stage3_artifact=base_stage3_artifact,
    )
    path = stage3_context_artifact_path(project_root=project_root, identity=identity)
    if not path.exists():
        return None, identity, path
    artifact = json.loads(path.read_text(encoding="utf-8"))
    stored = artifact.get("artifact_sha256")
    unhashed = {key: value for key, value in artifact.items() if key != "artifact_sha256"}
    if not stored or stored != _sha256_json(unhashed):
        raise Stage3ContextArtifactError("Frozen contextual Stage 3 artifact hash mismatch")
    for key, expected in identity.items():
        if artifact.get(key) != expected:
            raise Stage3ContextArtifactError(f"Frozen contextual Stage 3 artifact identity mismatch for {key}")
    bundle = Stage3ContextualDecisionBundle.model_validate(artifact.get("contextual_decision_bundle"))
    if bundle.scenario_set_version != identity["scenario_set_version"]:
        raise Stage3ContextArtifactError("Frozen contextual Stage 3 scenario-set mismatch")
    if artifact.get("stage3_contextual_output_sha256") != _sha256_json(bundle.model_dump(mode="json")):
        raise Stage3ContextArtifactError("Frozen contextual Stage 3 output hash mismatch")
    return artifact, identity, path


def freeze_stage3_context_artifact(
    *,
    project_root: str | Path,
    evidence_pack: dict[str, Any],
    base_stage3_artifact: dict[str, Any],
    contextual_decision_bundle: Stage3ContextualDecisionBundle | dict[str, Any],
) -> tuple[dict[str, Any], Path]:
    identity = build_stage3_context_identity(
        evidence_pack=evidence_pack,
        base_stage3_artifact=base_stage3_artifact,
    )
    bundle = (
        contextual_decision_bundle
        if isinstance(contextual_decision_bundle, Stage3ContextualDecisionBundle)
        else Stage3ContextualDecisionBundle.model_validate(contextual_decision_bundle)
    )
    if bundle.case_id != identity["case_id"]:
        raise Stage3ContextArtifactError("Contextual Stage 3 bundle case mismatch")
    if bundle.base_stage3_input_fingerprint != identity["base_stage3_input_fingerprint"]:
        raise Stage3ContextArtifactError("Contextual Stage 3 base fingerprint mismatch")
    if bundle.scenario_set_version != identity["scenario_set_version"]:
        raise Stage3ContextArtifactError("Contextual Stage 3 bundle scenario-set mismatch")
    output = bundle.model_dump(mode="json")
    artifact: dict[str, Any] = {
        **identity,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage3_contextual_output_sha256": _sha256_json(output),
        "contextual_decision_bundle": output,
    }
    artifact["artifact_sha256"] = _sha256_json(artifact)
    path = stage3_context_artifact_path(project_root=project_root, identity=identity)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(artifact, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError:
        existing, _, _ = load_frozen_stage3_context_artifact(
            project_root=project_root,
            evidence_pack=evidence_pack,
            base_stage3_artifact=base_stage3_artifact,
        )
        if existing is None or existing.get("stage3_contextual_output_sha256") != artifact["stage3_contextual_output_sha256"]:
            raise Stage3ContextArtifactError(
                "A contextual Stage 3 artifact already exists for this exact identity with different output"
            )
        return existing, path
    return artifact, path

