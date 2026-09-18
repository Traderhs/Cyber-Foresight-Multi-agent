from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from Stage7.config import (
    STAGE7_ARTIFACT_SCHEMA_VERSION,
    STAGE7_EXPERIMENT_MANIFEST_VERSION,
    STAGE7_HARNESS_VERSION,
    experiment_manifest,
)
from Stage7.loaders import CaseBundle, sha256_json
from Stage7.runtime import get_stage7_runtime_profile
from Stage7.schema import AxisResult, Stage7ValidationArtifact


def ensure_experiment_manifest(root: Path) -> tuple[dict[str, Any], Path, str]:
    manifest = experiment_manifest()
    path = (
        root
        / "Multi-Agent"
        / "Results"
        / "Stage7"
        / "manifests"
        / f"{STAGE7_EXPERIMENT_MANIFEST_VERSION}.json"
    )
    serialized = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != serialized:
        raise ValueError(
            f"Frozen Stage 7 experiment manifest changed in place: {path}. "
            "Bump the manifest version instead of overwriting the experiment definition."
        )
    if not path.exists():
        path.write_text(serialized, encoding="utf-8", newline="\n")
    return manifest, path, sha256_json(manifest)


def stage7_base_identity(
    bundles: list[CaseBundle],
    *,
    experiment_manifest_sha256: str,
) -> dict[str, Any]:
    identity = {
        "schema_version": STAGE7_ARTIFACT_SCHEMA_VERSION,
        "harness_version": STAGE7_HARNESS_VERSION,
        "experiment_manifest_version": STAGE7_EXPERIMENT_MANIFEST_VERSION,
        "experiment_manifest_sha256": experiment_manifest_sha256,
        "source_stage6_artifact_sha256": {
            bundle.case_id: str(bundle.stage6["artifact_sha256"])
            for bundle in bundles
        },
        "runtime": get_stage7_runtime_profile(),
    }
    return identity


def _evaluation_state(
    *,
    axes: list[AxisResult],
    grounding_audits: list[dict[str, Any]],
    mediator_audits: list[dict[str, Any]],
    manual_audit_sample_path: Path | None,
    mediator_audit_sample_path: Path | None,
) -> dict[str, Any]:
    return {
        "axes": [axis.model_dump(mode="json") for axis in axes],
        "grounding_audits": grounding_audits,
        "mediator_audits": mediator_audits,
        "manual_audit_sample_path": (
            str(manual_audit_sample_path) if manual_audit_sample_path else None
        ),
        "mediator_audit_sample_path": (
            str(mediator_audit_sample_path) if mediator_audit_sample_path else None
        ),
    }


def stage7_artifact_identity(
    bundles: list[CaseBundle],
    *,
    experiment_manifest_sha256: str,
    axes: list[AxisResult],
    grounding_audits: list[dict[str, Any]],
    mediator_audits: list[dict[str, Any]],
    manual_audit_sample_path: Path | None,
    mediator_audit_sample_path: Path | None,
) -> dict[str, Any]:
    identity = stage7_base_identity(
        bundles,
        experiment_manifest_sha256=experiment_manifest_sha256,
    )
    evaluation_state = _evaluation_state(
        axes=axes,
        grounding_audits=grounding_audits,
        mediator_audits=mediator_audits,
        manual_audit_sample_path=manual_audit_sample_path,
        mediator_audit_sample_path=mediator_audit_sample_path,
    )
    identity["evaluation_state_sha256"] = sha256_json(evaluation_state)
    identity["input_fingerprint"] = sha256_json(identity)
    return identity


def stage7_artifact_path(root: Path, identity: dict[str, Any]) -> Path:
    return (
        root
        / "Multi-Agent"
        / "Results"
        / "Stage7"
        / "artifacts"
        / f"{identity['input_fingerprint']}.json"
    )


def freeze_stage7_artifact(
    root: Path,
    *,
    bundles: list[CaseBundle],
    axes: list[AxisResult],
    grounding_audits: list[dict[str, Any]],
    mediator_audits: list[dict[str, Any]],
    manual_audit_sample_path: Path | None,
    mediator_audit_sample_path: Path | None,
) -> tuple[dict[str, Any], Path]:
    _, _, manifest_sha = ensure_experiment_manifest(root)
    identity = stage7_artifact_identity(
        bundles,
        experiment_manifest_sha256=manifest_sha,
        axes=axes,
        grounding_audits=grounding_audits,
        mediator_audits=mediator_audits,
        manual_audit_sample_path=manual_audit_sample_path,
        mediator_audit_sample_path=mediator_audit_sample_path,
    )
    artifact = Stage7ValidationArtifact(
        schema_version=STAGE7_ARTIFACT_SCHEMA_VERSION,
        harness_version=STAGE7_HARNESS_VERSION,
        experiment_manifest_version=STAGE7_EXPERIMENT_MANIFEST_VERSION,
        created_at=datetime.now(timezone.utc).isoformat(),
        input_fingerprint=identity["input_fingerprint"],
        main_case_count=len(bundles),
        main_scenario_count=sum(len(bundle.syntheses) for bundle in bundles),
        source_stage6_artifact_sha256=identity["source_stage6_artifact_sha256"],
        experiment_manifest_sha256=manifest_sha,
        evaluation_state_sha256=identity["evaluation_state_sha256"],
        runtime=identity["runtime"],
        axes=axes,
        grounding_audits=grounding_audits,
        mediator_audits=mediator_audits,
        manual_audit_sample_path=(
            str(manual_audit_sample_path) if manual_audit_sample_path else None
        ),
        mediator_audit_sample_path=(
            str(mediator_audit_sample_path) if mediator_audit_sample_path else None
        ),
    ).model_dump(mode="json")
    artifact["artifact_sha256"] = sha256_json(artifact)
    path = stage7_artifact_path(root, identity)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        old = {key: value for key, value in existing.items() if key not in {"artifact_sha256", "created_at"}}
        new = {key: value for key, value in artifact.items() if key not in {"artifact_sha256", "created_at"}}
        if old != new:
            raise ValueError(
                f"Conflicting Stage 7 artifact for exact frozen validation identity: {path}"
            )
        return existing, path
    path.write_text(serialized, encoding="utf-8", newline="\n")
    return artifact, path
