from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from Stage3.schema import (
    STAGE3_SELECTION_RULE_VERSION,
    STAGE3_SEMANTIC_VALIDATION_VERSION,
    Stage3DecisionBundle,
)


STAGE3_ARTIFACT_SCHEMA_VERSION = "stage3-artifact-v2"


class Stage3ArtifactError(ValueError):
    """Raised when a frozen Stage 3 artifact violates its deterministic contract."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _safe_case_id(case_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", case_id).strip("._")
    return safe or "case"


def _normalized_stage3_context(stage0_config: dict[str, Any]) -> dict[str, Any]:
    context = {
        "threat": stage0_config.get("threat"),
        "pmt": stage0_config.get("pmt"),
    }
    for key in ("region", "organization_type", "infrastructure_context", "budget_context"):
        value = stage0_config.get(key)
        context[key] = value if value not in (None, "") else None
    if context["threat"] in (None, "") or context["pmt"] in (None, ""):
        raise Stage3ArtifactError("Stage 3 artifact identity requires explicit threat and pmt names")
    return context


def _validate_bundle_binding(
    *,
    bundle: Stage3DecisionBundle,
    evidence_pack: dict[str, Any],
    identity: dict[str, Any],
) -> None:
    if bundle.case_id != identity["case_id"]:
        raise Stage3ArtifactError("Frozen Stage 3 decision bundle case_id mismatch")
    if bundle.candidate_selection_rule != identity["candidate_selection_rule"]:
        raise Stage3ArtifactError("Frozen Stage 3 decision bundle selection rule mismatch")
    expected_pmt_id = str(evidence_pack.get("pmt_id") or "")
    expected_threat_id = str(evidence_pack.get("threat_id") or "")
    action_ids = [item.action_id for item in bundle.eligible_action_set]
    if action_ids != [expected_pmt_id]:
        raise Stage3ArtifactError(
            f"Stage 3 v1 eligible action must equal the frozen case PMT: expected={expected_pmt_id!r}, "
            f"actual={action_ids!r}"
        )
    for decision_object in bundle.decision_objects:
        if decision_object.pmt_or_mitigation_id != expected_pmt_id:
            raise Stage3ArtifactError("Stage 3 DecisionObject PMT does not match the frozen EvidencePack PMT")
        if decision_object.threat_id != expected_threat_id:
            raise Stage3ArtifactError("Stage 3 DecisionObject Threat does not match the frozen EvidencePack Threat")


def _validate_stage2_artifact_binding(
    *,
    stage2_artifact: dict[str, Any],
    expected_case_id: str,
) -> None:
    stored_hash = str(stage2_artifact.get("artifact_sha256") or "")
    output_hash = str(stage2_artifact.get("stage2_output_sha256") or "")
    debate_result = stage2_artifact.get("debate_result")
    if not stored_hash or not output_hash or debate_result is None:
        raise Stage3ArtifactError("Stage 3 requires a complete frozen Stage 2 artifact")
    unhashed = {key: value for key, value in stage2_artifact.items() if key != "artifact_sha256"}
    if stored_hash != _sha256_json(unhashed):
        raise Stage3ArtifactError("Stage 3 rejected a Stage 2 artifact with an invalid content hash")
    if output_hash != _sha256_json(debate_result):
        raise Stage3ArtifactError("Stage 3 rejected a Stage 2 artifact with an invalid output hash")
    stage2_case_id = str(stage2_artifact.get("case_id") or debate_result.get("case_id") or "")
    if stage2_case_id != expected_case_id:
        raise Stage3ArtifactError(
            f"Stage 2/Stage 0 case_id mismatch at Stage 3 artifact boundary: "
            f"stage2={stage2_case_id!r}, stage0={expected_case_id!r}"
        )


def build_stage3_identity(
    *,
    evidence_pack: dict[str, Any],
    stage2_artifact: dict[str, Any],
    stage0_config: dict[str, Any],
) -> dict[str, Any]:
    case_id = str(evidence_pack.get("case_id") or "")
    if not case_id:
        raise Stage3ArtifactError("Stage 3 requires a Stage 0 case_id")
    _validate_stage2_artifact_binding(stage2_artifact=stage2_artifact, expected_case_id=case_id)
    stage2_hash = str(stage2_artifact["artifact_sha256"])
    stage2_output_hash = str(stage2_artifact["stage2_output_sha256"])
    stage3_context = _normalized_stage3_context(stage0_config)
    identity: dict[str, Any] = {
        "schema_version": STAGE3_ARTIFACT_SCHEMA_VERSION,
        "semantic_validation_version": STAGE3_SEMANTIC_VALIDATION_VERSION,
        "candidate_selection_rule": STAGE3_SELECTION_RULE_VERSION,
        "case_id": case_id,
        "stage0_evidence_pack_sha256": _sha256_json(evidence_pack),
        "stage2_artifact_sha256": stage2_hash,
        "stage2_output_sha256": stage2_output_hash,
        "stage3_context": stage3_context,
        "decision_bundle_schema_sha256": _sha256_json(Stage3DecisionBundle.model_json_schema()),
    }
    identity["input_fingerprint"] = _sha256_json(identity)
    return identity


def stage3_artifact_path(*, project_root: str | Path, identity: dict[str, Any]) -> Path:
    return (
        Path(project_root).resolve()
        / "Multi-Agent"
        / "Results"
        / "Stage3"
        / "artifacts"
        / _safe_case_id(str(identity["case_id"]))
        / f"{identity['input_fingerprint']}.json"
    )


def load_frozen_stage3_artifact(
    *,
    project_root: str | Path,
    evidence_pack: dict[str, Any],
    stage2_artifact: dict[str, Any],
    stage0_config: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any], Path]:
    identity = build_stage3_identity(
        evidence_pack=evidence_pack,
        stage2_artifact=stage2_artifact,
        stage0_config=stage0_config,
    )
    path = stage3_artifact_path(project_root=project_root, identity=identity)
    if not path.exists():
        return None, identity, path
    artifact = json.loads(path.read_text(encoding="utf-8"))
    stored_hash = artifact.get("artifact_sha256")
    unhashed = {key: value for key, value in artifact.items() if key != "artifact_sha256"}
    if not stored_hash or stored_hash != _sha256_json(unhashed):
        raise Stage3ArtifactError("Frozen Stage 3 artifact hash mismatch")
    for key, expected in identity.items():
        if artifact.get(key) != expected:
            raise Stage3ArtifactError(f"Frozen Stage 3 artifact identity mismatch for {key}")
    bundle = Stage3DecisionBundle.model_validate(artifact.get("decision_bundle"))
    _validate_bundle_binding(bundle=bundle, evidence_pack=evidence_pack, identity=identity)
    if artifact.get("stage3_output_sha256") != _sha256_json(bundle.model_dump(mode="json")):
        raise Stage3ArtifactError("Frozen Stage 3 output hash mismatch")
    return artifact, identity, path


def freeze_stage3_artifact(
    *,
    project_root: str | Path,
    evidence_pack: dict[str, Any],
    stage2_artifact: dict[str, Any],
    stage0_config: dict[str, Any],
    decision_bundle: Stage3DecisionBundle | dict[str, Any],
) -> tuple[dict[str, Any], Path]:
    identity = build_stage3_identity(
        evidence_pack=evidence_pack,
        stage2_artifact=stage2_artifact,
        stage0_config=stage0_config,
    )
    bundle = (
        decision_bundle
        if isinstance(decision_bundle, Stage3DecisionBundle)
        else Stage3DecisionBundle.model_validate(decision_bundle)
    )
    _validate_bundle_binding(bundle=bundle, evidence_pack=evidence_pack, identity=identity)
    output = bundle.model_dump(mode="json")
    artifact: dict[str, Any] = {
        **identity,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_snapshot_id": evidence_pack.get("source_snapshot_id"),
        "stage3_output_sha256": _sha256_json(output),
        "decision_bundle": output,
    }
    artifact["artifact_sha256"] = _sha256_json(artifact)
    path = stage3_artifact_path(project_root=project_root, identity=identity)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(artifact, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError:
        existing, _, _ = load_frozen_stage3_artifact(
            project_root=project_root,
            evidence_pack=evidence_pack,
            stage2_artifact=stage2_artifact,
            stage0_config=stage0_config,
        )
        if existing is None or existing.get("stage3_output_sha256") != artifact["stage3_output_sha256"]:
            raise Stage3ArtifactError(
                "A Stage 3 artifact already exists for this exact identity with a different output"
            )
        artifact = existing
    return artifact, path
