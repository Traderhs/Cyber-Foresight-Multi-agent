from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

from .bm25 import (
    BM25_B,
    BM25_K1,
    BM25_RETRIEVER_VERSION,
    build_slot_query,
    rank_bm25,
)
from .registry import SOURCE_REGISTRY_VERSION, registry_by_id, validate_source_registry
from .schema import EvidenceRecord, Stage0ValidationError


DEFAULT_REQUIRED_EVIDENCE_SLOTS: tuple[str, ...] = (
    "observed_threat_reality",
    "threat_capability_or_mechanism",
    "exploitation_signal",
    "mitigation_relation_or_control",
    "mitigation_implementation",
    "technical_literature",
)

MAX_RECORDS_PER_EVIDENCE_SLOT = 4

_EVIDENCE_SLOT_CLAIM_TYPES: dict[str, frozenset[str]] = {
    "observed_threat_reality": frozenset(
        {
            "observed_threat_pattern",
            "directional_threat_trend",
            "observed_attack_vector",
            "observed_adversary_behavior",
            "identity_cloud_ai_threat_trend",
            "campaign_evidence",
            "critical_infrastructure_risk_framing",
            "synthetic_content_risk",
        }
    ),
    "threat_capability_or_mechanism": frozenset(
        {
            "observed_ttp",
            "adversary_tactic",
            "adversary_technique",
            "adversary_procedure",
            "malware_behavior",
            "adversarial_ml_mechanism",
            "ai_attack_technique",
            "ai_case_mapping",
            "deepfake_threat_mechanism",
            "information_manipulation_mechanism",
            "iot_security_capability",
        }
    ),
    "exploitation_signal": frozenset(
        {
            "confirmed_exploitation",
            "exploitation_probability",
            "vulnerability_identity",
            "affected_product_version",
            "vulnerability_severity",
            "vulnerability_enrichment",
            "cwe_mapping",
            "cpe_mapping",
        }
    ),
    "mitigation_relation_or_control": frozenset(
        {
            "attack_defense_relation",
            "defensive_technique_mapping",
            "official_mitigation",
            "ai_mitigation_class",
            "deepfake_detection_class",
            "deepfake_mitigation_guidance",
            "insider_threat_control",
            "insider_detection_mitigation",
            "supply_chain_risk_control",
            "recommended_control",
            "prioritized_defensive_practice",
            "security_control",
            "privacy_control",
            "korea_control_requirement",
        }
    ),
    "mitigation_implementation": frozenset(
        {
            "organizational_applicability",
            "organizational_security_outcome",
            "cyber_governance",
            "implementation_feasibility",
            "reference_architecture",
            "implementation_how_to",
            "control_assessment_procedure",
            "implementation_assessment",
            "iot_manufacturer_support_baseline",
            "sbom_practice",
            "software_dependency_transparency",
            "supply_chain_governance",
            "legal_requirement",
            "regulatory_scope",
            "eu_legal_requirement",
            "us_legal_requirement",
        }
    ),
    "technical_literature": frozenset(
        {
            "scholarly_work_discovery",
            "preprint_metadata",
            "academic_topic_metadata",
            "doi_verification",
            "publisher_metadata",
            "publication_metadata",
            "directional_publication_trend",
        }
    ),
}

def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _dump_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class EvidenceSnapshotWriter:
    """Builds a new immutable evidence snapshot from the fixed source registry.

    There is deliberately no API for inserting Agent reasoning.
    """

    def __init__(self, evidence_root: str | Path, snapshot_id: str):
        self.root = Path(evidence_root).resolve()
        self.snapshot_id = snapshot_id
        self.snapshot_dir = self.root / "snapshots" / snapshot_id
        self.artifact_dir = self.snapshot_dir / "artifacts"
        self.records: list[EvidenceRecord] = []
        self._evidence_ids: set[str] = set()
        self.artifacts: list[dict[str, Any]] = []
        self.audit_metadata: dict[str, Any] = {}
        self._finalized = False
        if self.snapshot_dir.exists():
            raise Stage0ValidationError(f"Evidence snapshot already exists and is immutable: {self.snapshot_dir}")
        validate_source_registry()

    def add_artifact(
        self,
        *,
        source_registry_id: str,
        source_path: str | Path,
        artifact_id: str,
        publication_date: str,
        available_at: str,
        version: str,
        parser_version: str = "raw-v1",
        source_uri: str | None = None,
    ) -> dict[str, Any]:
        self._ensure_mutable()
        registry = registry_by_id()
        if source_registry_id not in registry:
            raise Stage0ValidationError(f"Unknown source_registry_id {source_registry_id}")
        self._validate_date(publication_date, "publication_date")
        self._validate_date(available_at, "available_at")
        path = Path(source_path).resolve()
        if not path.is_file():
            raise Stage0ValidationError(f"Artifact does not exist: {path}")
        digest = _sha256_file(path)
        extension = path.suffix.lower()
        destination_name = f"{source_registry_id}_{artifact_id}_{digest[:12]}{extension}"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        destination = self.artifact_dir / destination_name
        shutil.copy2(path, destination)
        entry = {
            "source_registry_id": source_registry_id,
            "artifact_id": artifact_id,
            "source_name": registry[source_registry_id].name,
            "original_name": path.name,
            "snapshot_path": str(destination.relative_to(self.snapshot_dir)).replace("\\", "/"),
            "publication_date": publication_date,
            "available_at": available_at,
            "version": version,
            "content_hash": digest,
            "parser_version": parser_version,
            "source_uri": source_uri,
        }
        self.artifacts.append(entry)
        return entry

    def set_audit_metadata(self, metadata: dict[str, Any]) -> None:
        self._ensure_mutable()
        self.audit_metadata = json.loads(json.dumps(metadata, ensure_ascii=False))

    def abort(self) -> None:
        """Remove a non-finalized snapshot after an ingestion failure."""
        if self._finalized:
            raise Stage0ValidationError("Cannot abort a finalized immutable snapshot")
        if self.snapshot_dir.exists():
            shutil.rmtree(self.snapshot_dir)

    def add_record(
        self,
        *,
        source_registry_id: str,
        source: str,
        publication_date: str,
        available_at: str,
        source_document_id: str,
        source_locator: str,
        evidence_type: str,
        content: str,
        evidence_date: str | None = None,
        evidence_chain_id: str | None = None,
        retrieval_role: str = "neutral_candidate",
        threat_ids: Iterable[str] = (),
        pmt_ids: Iterable[str] = (),
    ) -> EvidenceRecord:
        self._ensure_mutable()
        registry = registry_by_id()
        definition = registry.get(source_registry_id)
        if definition is None:
            raise Stage0ValidationError(f"Unknown source_registry_id {source_registry_id}")
        if not content.strip():
            raise Stage0ValidationError("Evidence content cannot be blank")
        self._validate_date(publication_date, "publication_date")
        self._validate_date(available_at, "available_at")
        if evidence_date:
            self._validate_date(evidence_date, "evidence_date")
        normalized = content.strip().encode("utf-8")
        digest = _sha256_bytes(normalized)
        evidence_id = f"E_{source_registry_id}_{digest[:16].upper()}"
        record = EvidenceRecord(
            evidence_id=evidence_id,
            source_registry_id=source_registry_id,
            source_family=definition.family,
            source=source,
            publication_date=publication_date,
            evidence_date=evidence_date,
            available_at=available_at,
            source_document_id=source_document_id,
            source_locator=source_locator,
            source_snapshot_id=self.snapshot_id,
            evidence_chain_id=evidence_chain_id,
            evidence_type=evidence_type,
            retrieval_role=retrieval_role,
            allowed_claim_types=definition.allowed_claim_types,
            prohibited_claim_types=definition.prohibited_claim_types,
            threat_ids=tuple(sorted(set(threat_ids))),
            pmt_ids=tuple(sorted(set(pmt_ids))),
            content=content.strip(),
            content_hash=digest,
        )
        record.validate()
        if record.evidence_id in self._evidence_ids:
            raise Stage0ValidationError(f"Duplicate evidence content/ID: {record.evidence_id}")
        self._evidence_ids.add(record.evidence_id)
        self.records.append(record)
        return record

    def finalize(self) -> Path:
        self._ensure_mutable()
        if (self.snapshot_dir / "manifest.json").exists():
            raise Stage0ValidationError(f"Evidence snapshot is already finalized: {self.snapshot_dir}")
        # add_artifact may already have created snapshot_dir/artifacts; finalize is still
        # allowed exactly once because immutability begins when manifest.json is written.
        self._write_files()
        self._finalized = True
        return self.snapshot_dir

    def _write_files(self) -> None:
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        registry_payload = {
            "source_registry_version": SOURCE_REGISTRY_VERSION,
            "validation": validate_source_registry(),
            "entries": [definition.to_dict() for definition in registry_by_id().values()],
        }
        _dump_json_atomic(self.root / "source_registry.json", registry_payload)
        records_path = self.snapshot_dir / "records.jsonl"
        with records_path.open("w", encoding="utf-8", newline="\n") as f:
            for record in sorted(self.records, key=lambda r: r.evidence_id):
                f.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
        manifest = {
            "source_registry_version": SOURCE_REGISTRY_VERSION,
            "source_snapshot_id": self.snapshot_id,
            "created_at": datetime.now().astimezone().isoformat(),
            "immutable": True,
            "artifact_count": len(self.artifacts),
            "record_count": len(self.records),
            "artifacts": sorted(self.artifacts, key=lambda a: (a["source_registry_id"], a["artifact_id"])),
            "records_sha256": _sha256_file(records_path),
            "audit": self.audit_metadata,
        }
        _dump_json_atomic(self.snapshot_dir / "manifest.json", manifest)

    def _ensure_mutable(self) -> None:
        if self._finalized:
            raise Stage0ValidationError("Evidence snapshot is already finalized and immutable")

    @staticmethod
    def _validate_date(value: str, name: str) -> None:
        try:
            date.fromisoformat(value)
        except ValueError as exc:
            raise Stage0ValidationError(f"{name} must be ISO YYYY-MM-DD, got {value!r}") from exc


class EvidenceStore:
    """Read-only access to one finalized Stage 0 evidence snapshot."""

    def __init__(self, snapshot_dir: str | Path):
        self.snapshot_dir = Path(snapshot_dir).resolve()
        self.manifest_path = self.snapshot_dir / "manifest.json"
        self.records_path = self.snapshot_dir / "records.jsonl"
        if not self.manifest_path.is_file() or not self.records_path.is_file():
            raise Stage0ValidationError(f"Not a finalized evidence snapshot: {self.snapshot_dir}")
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("immutable") is not True:
            raise Stage0ValidationError("Evidence snapshot manifest must declare immutable=true")
        if _sha256_file(self.records_path) != self.manifest.get("records_sha256"):
            raise Stage0ValidationError("Evidence records hash does not match snapshot manifest")
        self.records = self._load_records()
        self._validate_snapshot()

    @property
    def snapshot_id(self) -> str:
        return self.manifest["source_snapshot_id"]

    def query(
        self,
        *,
        cutoff_date: str,
        threat_id: str,
        pmt_id: str,
        threat_name: str,
        pmt_name: str,
        gap_direction: str,
        required_evidence_slots: Iterable[str] | None = None,
    ) -> tuple[list[EvidenceRecord], dict[str, Any]]:
        cutoff = date.fromisoformat(cutoff_date)
        required = tuple(required_evidence_slots or DEFAULT_REQUIRED_EVIDENCE_SLOTS)
        unknown_slots = sorted(set(required) - set(_EVIDENCE_SLOT_CLAIM_TYPES))
        if unknown_slots:
            raise Stage0ValidationError(f"Unknown required evidence slots: {unknown_slots}")
        temporal_violations = 0
        candidates: list[EvidenceRecord] = []
        for record in self.records:
            # Untagged records remain available in the frozen corpus for audit and
            # later context-specific retrieval, but are never treated as globally
            # relevant to every threat/PMT case.
            if not record.threat_ids and not record.pmt_ids:
                continue
            threat_match = not record.threat_ids or threat_id in record.threat_ids
            pmt_match = not record.pmt_ids or pmt_id in record.pmt_ids
            if not (threat_match and pmt_match):
                continue
            if date.fromisoformat(record.available_at) > cutoff:
                temporal_violations += 1
                continue
            candidates.append(record)

        slot_matches: dict[str, list[EvidenceRecord]] = {slot: [] for slot in required}
        for record in candidates:
            allowed = set(record.allowed_claim_types)
            matched = tuple(
                slot
                for slot in required
                if allowed & _EVIDENCE_SLOT_CLAIM_TYPES[slot]
            )
            for slot in matched:
                slot_matches[slot].append(record)

        selected_by_id: dict[str, EvidenceRecord] = {}
        selected_slots: dict[str, set[str]] = {}
        retrieval_queries: dict[str, str] = {}
        bm25_scores: dict[str, dict[str, float]] = {}
        slot_candidate_counts: dict[str, int] = {}
        for slot in required:
            slot_candidates = slot_matches[slot]
            slot_candidate_counts[slot] = len(slot_candidates)
            query_text = build_slot_query(
                threat_name=threat_name,
                pmt_name=pmt_name,
                gap_direction=gap_direction,
                slot=slot,
            )
            retrieval_queries[slot] = query_text
            scored = rank_bm25(
                [record.content for record in slot_candidates],
                query_text,
            )
            score_by_id = {
                slot_candidates[result.index].evidence_id: result.score
                for result in scored
            }
            bm25_scores[slot] = {
                evidence_id: round(score, 12)
                for evidence_id, score in sorted(score_by_id.items())
            }
            ranked = sorted(
                slot_candidates,
                key=lambda r: (
                    -score_by_id[r.evidence_id],
                    -(int(threat_id in r.threat_ids) + int(pmt_id in r.pmt_ids)),
                    r.source_family,
                    r.source_registry_id,
                    r.source_document_id,
                    r.evidence_id,
                ),
            )
            chosen: list[EvidenceRecord] = []
            used_sources: set[str] = set()
            for record in ranked:
                if record.source_registry_id in used_sources:
                    continue
                chosen.append(record)
                used_sources.add(record.source_registry_id)
                if len(chosen) == MAX_RECORDS_PER_EVIDENCE_SLOT:
                    break
            if len(chosen) < MAX_RECORDS_PER_EVIDENCE_SLOT:
                for record in ranked:
                    if record in chosen:
                        continue
                    chosen.append(record)
                    if len(chosen) == MAX_RECORDS_PER_EVIDENCE_SLOT:
                        break
            for record in chosen:
                selected_by_id[record.evidence_id] = record
                selected_slots.setdefault(record.evidence_id, set()).add(slot)

        selected = sorted(
            selected_by_id.values(),
            key=lambda r: (
                -(int(threat_id in r.threat_ids) + int(pmt_id in r.pmt_ids)),
                r.source_family,
                r.source_registry_id,
                r.evidence_id,
            ),
        )
        covered = [slot for slot in required if slot_matches[slot]]
        chains = {
            r.evidence_chain_id or f"{r.source_registry_id}:{r.source_document_id}"
            for r in selected
        }
        metadata = {
            "cutoff_applied": cutoff_date,
            "candidate_count_before_selection": len(candidates),
            "retrieved_count": len(selected),
            "retrieval_method": "hard_filter_then_okapi_bm25_then_source_diversity_top_k",
            "retriever_version": BM25_RETRIEVER_VERSION,
            "bm25_parameters": {"k1": BM25_K1, "b": BM25_B},
            "retrieval_queries": retrieval_queries,
            "slot_candidate_counts": slot_candidate_counts,
            "bm25_scores_by_slot": bm25_scores,
            "max_records_per_evidence_slot": MAX_RECORDS_PER_EVIDENCE_SLOT,
            "source_family_count": len({r.source_family for r in selected}),
            "independent_evidence_chain_count": len(chains),
            "required_evidence_slots": list(required),
            "covered_evidence_slots": covered,
            "missing_evidence_slots": sorted(set(required) - set(covered)),
            "evidence_slot_coverage": {
                evidence_id: sorted(slots)
                for evidence_id, slots in sorted(selected_slots.items())
            },
            "retrieval_intents": ["support_candidate", "contradiction_candidate"],
            "semantic_stance_assignment": "downstream_critic_not_stage0",
            "temporal_violation_count": 0,
            "excluded_post_cutoff_record_count": temporal_violations,
            "source_contract_violation_count": 0,
            "evidence_sufficiency": (
                "SUFFICIENT"
                if required and set(required).issubset(covered)
                else "PARTIAL"
                if covered
                else "INSUFFICIENT_EVIDENCE"
            ),
        }
        return selected, metadata

    def _load_records(self) -> list[EvidenceRecord]:
        records: list[EvidenceRecord] = []
        with self.records_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = EvidenceRecord.from_dict(json.loads(line))
                record.validate()
                records.append(record)
        return records

    def _validate_snapshot(self) -> None:
        if self.manifest.get("source_registry_version") != SOURCE_REGISTRY_VERSION:
            raise Stage0ValidationError(
                f"Evidence snapshot registry version {self.manifest.get('source_registry_version')} != runtime {SOURCE_REGISTRY_VERSION}"
            )
        ids: set[str] = set()
        registry = registry_by_id()
        for record in self.records:
            if record.evidence_id in ids:
                raise Stage0ValidationError(f"Duplicate evidence ID {record.evidence_id}")
            ids.add(record.evidence_id)
            definition = registry.get(record.source_registry_id)
            if definition is None:
                raise Stage0ValidationError(f"Evidence references unknown registry source {record.source_registry_id}")
            if record.source_family != definition.family:
                raise Stage0ValidationError(f"Source family mismatch for {record.evidence_id}")
            if record.allowed_claim_types != definition.allowed_claim_types:
                raise Stage0ValidationError(f"Allowed claim contract drift for {record.evidence_id}")
            if record.prohibited_claim_types != definition.prohibited_claim_types:
                raise Stage0ValidationError(f"Prohibited claim contract drift for {record.evidence_id}")
            if hashlib.sha256(record.content.encode("utf-8")).hexdigest() != record.content_hash:
                raise Stage0ValidationError(f"Evidence content hash mismatch for {record.evidence_id}")
        if len(records := self.records) != self.manifest.get("record_count"):
            raise Stage0ValidationError(f"Manifest record_count={self.manifest.get('record_count')} but loaded {len(records)}")
        artifacts = self.manifest.get("artifacts") or []
        if len(artifacts) != self.manifest.get("artifact_count"):
            raise Stage0ValidationError(
                f"Manifest artifact_count={self.manifest.get('artifact_count')} but lists {len(artifacts)} artifacts"
            )
        for artifact in artifacts:
            relative = artifact.get("snapshot_path")
            expected_hash = artifact.get("content_hash")
            if not relative or not expected_hash:
                raise Stage0ValidationError("Evidence artifact manifest entry is missing snapshot_path/content_hash")
            artifact_path = self.snapshot_dir / relative
            if not artifact_path.is_file():
                raise Stage0ValidationError(f"Evidence artifact is missing: {relative}")
            if _sha256_file(artifact_path) != expected_hash:
                raise Stage0ValidationError(f"Evidence artifact hash mismatch: {relative}")

