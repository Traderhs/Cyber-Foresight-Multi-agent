from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Any, Iterable

from .evidence_store import EvidenceStore
from .registry import SOURCE_REGISTRY_VERSION
from .schema import (
    BasisType,
    ClaimReference,
    ClaimReferenceAudit,
    ClaimReferenceStatus,
    EvaluationMode,
    EvidencePack,
    Stage0ValidationError,
)


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


class Stage0Builder:
    """Build EvidencePack objects from the paper-native forecast contract and frozen evidence only."""

    def __init__(self, forecast_dir: str | Path, evidence_snapshot_dir: str | Path):
        self.forecast_dir = Path(forecast_dir).resolve()
        self.evidence_store = EvidenceStore(evidence_snapshot_dir)
        self.manifest = self._load_json("manifest.json")
        self.entities = self._load_json("node_registry.json")
        self.feature_contract = self._load_json("feature_contract.json")
        self.state_contract = self._load_json("node_state_contract.json")
        self.gaps = self._load_jsonl("threat_pmt_gaps.jsonl")
        self.series = self._load_feature_series()
        self.state_series = self._load_state_series()
        self._validate_canonical_contract()

    def build(
        self,
        *,
        case_id: str,
        threat: str,
        pmt: str,
        analysis_cutoff_date: str,
        evaluation_mode: EvaluationMode | str,
        required_evidence_slots: Iterable[str] | None = None,
    ) -> EvidencePack:
        mode = EvaluationMode(evaluation_mode)
        cutoff = date.fromisoformat(analysis_cutoff_date)
        provenance = self.manifest["forecast_provenance"]
        forecast_origin_date = provenance["forecast_origin_date"]
        origin = date.fromisoformat(forecast_origin_date)
        if mode == EvaluationMode.EX_ANTE_REPLAY and cutoff > origin:
            raise Stage0ValidationError(
                f"Ex-ante replay cutoff {cutoff} cannot be later than forecast origin {origin}"
            )
        threat_entity = self._find_entity("threat", threat)
        pmt_entity = self._find_entity("pmt", pmt)
        gap_rows = [
            row
            for row in self.gaps
            if row["threat_id"] == threat_entity["node_id"] and row["pmt_id"] == pmt_entity["node_id"]
        ]
        if not gap_rows:
            raise Stage0ValidationError(
                f"No paper-contract gap exists for threat={threat!r}, pmt={pmt!r}; Stage 0 will not invent one at runtime"
            )
        gap_rows.sort(key=lambda r: r["year"])
        gaps = [row["gap"] for row in gap_rows]
        gap_slope = (gaps[-1] - gaps[0]) / (gap_rows[-1]["year"] - gap_rows[0]["year"])
        gap_direction = (
            "threat_minus_pmt_increasing"
            if gaps[-1] > gaps[0]
            else "threat_minus_pmt_decreasing"
            if gaps[-1] < gaps[0]
            else "flat"
        )
        threat_features = self._feature_summary(threat_entity["node_id"])
        pmt_features = self._feature_summary(pmt_entity["node_id"])
        threat_state = self._state_summary(threat_entity["node_id"])
        pmt_state = self._state_summary(pmt_entity["node_id"])
        evidence, retrieval_metadata = self.evidence_store.query(
            cutoff_date=analysis_cutoff_date,
            threat_id=threat_entity["node_id"],
            pmt_id=pmt_entity["node_id"],
            threat_name=threat_entity["canonical_name"],
            pmt_name=pmt_entity["canonical_name"],
            gap_direction=gap_direction,
            required_evidence_slots=required_evidence_slots,
        )
        retrieval_metadata["query"] = {
            "threat_id": threat_entity["node_id"],
            "pmt_id": pmt_entity["node_id"],
            "gap_direction": gap_direction,
            "forecast_horizon": [2025, 2027],
            "analysis_cutoff_date": analysis_cutoff_date,
            "required_evidence_slots": retrieval_metadata["required_evidence_slots"],
            "retrieval_intents": retrieval_metadata["retrieval_intents"],
        }
        uncertainty = {
            "status": self.manifest["uncertainty_semantics"]["paper_status"],
            "threat_confidence_95_half_width_z_mean": threat_state["confidence_95_half_width_z_mean"],
            "pmt_confidence_95_half_width_z_mean": pmt_state["confidence_95_half_width_z_mean"],
            "legacy_variance_warning": self.manifest["uncertainty_semantics"]["legacy_variance"],
        }
        pack = EvidencePack(
            case_id=case_id,
            source_registry_version=SOURCE_REGISTRY_VERSION,
            source_snapshot_id=self.evidence_store.snapshot_id,
            forecast_origin_date=forecast_origin_date,
            training_data_end=provenance["training_data_end"],
            analysis_cutoff_date=analysis_cutoff_date,
            evaluation_mode=mode,
            threat_id=threat_entity["node_id"],
            pmt_id=pmt_entity["node_id"],
            forecast_summary={
                "paper_feature_order": self.manifest["paper_contract"]["feature_names"],
                "historical_input": {
                    "threat_features": threat_features,
                    "pmt_features": pmt_features,
                },
                "threat_state": threat_state,
                "pmt_state": pmt_state,
                "gap_by_year": {str(row["year"]): row["gap"] for row in gap_rows},
                "gap_slope_per_year": gap_slope,
                "gap_direction": gap_direction,
                "gap_semantics": gap_rows[0]["value_semantics"],
                "gap_direction_semantics": (
                    "gap = threat_state_mean_z - pmt_state_mean_z; gap_direction describes signed gap movement, "
                    "not absolute-distance widening/narrowing"
                ),
                "predictive_uncertainty": uncertainty,
                "historical_field_ids": {
                    "threat": [field_id for feature in threat_features.values() for field_id in feature["historical_field_ids"]],
                    "pmt": [field_id for feature in pmt_features.values() for field_id in feature["historical_field_ids"]],
                },
                "forecast_field_ids": {
                    "threat": threat_state["forecast_field_ids"],
                    "pmt": pmt_state["forecast_field_ids"],
                    "gap": [row["gap_field_id"] for row in gap_rows],
                },
            },
            evidence=tuple(evidence),
            retrieval_metadata=retrieval_metadata,
        )
        pack.validate()
        return pack

    def audit_claim_reference(self, claim: ClaimReference, pack: EvidencePack) -> ClaimReferenceAudit:
        evidence_by_id = {record.evidence_id: record for record in pack.evidence}
        forecast_ids = set(pack.forecast_summary["forecast_field_ids"]["threat"])
        forecast_ids |= set(pack.forecast_summary["forecast_field_ids"]["pmt"])
        forecast_ids |= set(pack.forecast_summary["forecast_field_ids"]["gap"])
        issues: list[str] = []
        if not claim.evidence_ids and not claim.forecast_field_ids:
            return ClaimReferenceAudit(
                claim_id=claim.claim_id,
                status=ClaimReferenceStatus.UNSUPPORTED_CLAIM,
                issues=("Claim has no evidence_id or forecast_field_id references",),
            )
        unknown_evidence = sorted(set(claim.evidence_ids) - set(evidence_by_id))
        unknown_forecast = sorted(set(claim.forecast_field_ids) - forecast_ids)
        if unknown_evidence or unknown_forecast:
            issues.extend([f"Unknown evidence IDs: {unknown_evidence}" if unknown_evidence else "", f"Unknown forecast IDs: {unknown_forecast}" if unknown_forecast else ""])
            return ClaimReferenceAudit(
                claim_id=claim.claim_id,
                status=ClaimReferenceStatus.UNKNOWN_REFERENCE,
                issues=tuple(i for i in issues if i),
            )
        if claim.basis_type == BasisType.FORECAST_ONLY and claim.evidence_ids:
            issues.append("FORECAST_ONLY claim cannot cite external evidence")
        if claim.basis_type == BasisType.EXTERNAL_EVIDENCE and claim.forecast_field_ids:
            issues.append("EXTERNAL_EVIDENCE claim cannot cite forecast fields")
        if claim.basis_type == BasisType.FORECAST_PLUS_EVIDENCE and (not claim.evidence_ids or not claim.forecast_field_ids):
            issues.append("FORECAST_PLUS_EVIDENCE requires both evidence and forecast references")
        for evidence_id in claim.evidence_ids:
            record = evidence_by_id[evidence_id]
            if date.fromisoformat(record.available_at) > date.fromisoformat(pack.analysis_cutoff_date):
                return ClaimReferenceAudit(
                    claim_id=claim.claim_id,
                    status=ClaimReferenceStatus.TEMPORAL_VIOLATION,
                    issues=(f"{evidence_id} is available after case cutoff",),
                )
            if claim.claim_type in record.prohibited_claim_types or claim.claim_type not in record.allowed_claim_types:
                issues.append(f"{evidence_id} source contract does not allow claim type {claim.claim_type!r}")
        if issues:
            return ClaimReferenceAudit(
                claim_id=claim.claim_id,
                status=ClaimReferenceStatus.SOURCE_CONTRACT_VIOLATION,
                issues=tuple(issues),
            )
        return ClaimReferenceAudit(claim_id=claim.claim_id, status=ClaimReferenceStatus.VALID, issues=())

    def _validate_canonical_contract(self) -> None:
        if self.manifest.get("schema_version") != "stage0-paper-forecast-v3":
            raise Stage0ValidationError(f"Unsupported canonical forecast schema {self.manifest.get('schema_version')}")
        paper = self.manifest.get("paper_contract", {})
        if paper.get("node_count") != 124 or paper.get("feature_dim") != 4:
            raise Stage0ValidationError(f"Paper forecast tensor contract is invalid: {paper}")
        if paper.get("feature_names") != ["NoI", "NoP", "ACA", "PH"]:
            raise Stage0ValidationError(f"Unexpected paper feature order: {paper.get('feature_names')}")
        if len(self.entities) != 124 or len(self.feature_contract) != 124 * 4 or len(self.state_contract) != 124:
            raise Stage0ValidationError("Canonical files disagree with paper node-feature contract")
        expected_keys = {
            (entity["node_id"], feature)
            for entity in self.entities
            for feature in paper["feature_names"]
        }
        if set(self.series) != expected_keys:
            raise Stage0ValidationError("Node-feature series do not cover the full 124 x 4 paper contract")
        if set(self.state_series) != {entity["node_id"] for entity in self.entities}:
            raise Stage0ValidationError("Node-state series do not cover the full 124-node paper forecast contract")

    def _find_entity(self, entity_type: str, name: str) -> dict[str, Any]:
        matches = [e for e in self.entities if e["node_type"] == entity_type and _norm(e["canonical_name"]) == _norm(name)]
        if len(matches) != 1:
            raise Stage0ValidationError(f"Entity lookup failed: type={entity_type}, name={name!r}, matches={len(matches)}")
        return matches[0]

    def _feature_summary(self, node_id: str) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        for feature in self.manifest["paper_contract"]["feature_names"]:
            rows = self.series[(node_id, feature)]
            available = any(row["available"] for row in rows)
            values = [row["value_z"] for row in rows if row["available"]]
            recent = values[-36:]
            summary[feature] = {
                "available": available,
                "recent_direction": self._direction(recent) if recent else "masked",
                "latest_value_z": values[-1] if values else None,
                "historical_field_ids": [row["field_id"] for row in rows],
            }
        return summary

    def _state_summary(self, node_id: str) -> dict[str, Any]:
        rows = self.state_series[node_id]
        forecast_rows = [row for row in rows if row["phase"] == "forecast"]
        if len(forecast_rows) != 36:
            raise Stage0ValidationError(f"Node {node_id} does not have a 36-month forecast")
        values = [row["value_z"] for row in forecast_rows]
        confidence = [row["confidence_95_half_width_z"] for row in forecast_rows]
        state_meta = next(item for item in self.state_contract if item["node_id"] == node_id)
        return {
            "state_modality": state_meta["state_modality"],
            "direction": self._direction(values),
            "forecast_mean_z": sum(values) / len(values),
            "confidence_95_half_width_z_mean": sum(confidence) / len(confidence),
            "forecast_field_ids": [row["state_field_id"] for row in forecast_rows],
        }

    @staticmethod
    def _direction(values: list[float]) -> str:
        if len(values) < 2:
            return "unknown"
        slope = (values[-1] - values[0]) / (len(values) - 1)
        scale = max(abs(v) for v in values) or 1.0
        if abs(slope) / scale < 1e-4:
            return "flat"
        return "increasing" if slope > 0 else "decreasing"

    def _load_json(self, name: str) -> Any:
        path = self.forecast_dir / name
        if not path.is_file():
            raise Stage0ValidationError(f"Missing canonical file {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _load_jsonl(self, name: str) -> list[dict[str, Any]]:
        path = self.forecast_dir / name
        if not path.is_file():
            raise Stage0ValidationError(f"Missing canonical file {path}")
        with path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def _load_feature_series(self) -> dict[tuple[str, str], list[dict[str, Any]]]:
        rows = self._load_jsonl("historical_node_features.jsonl")
        series: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in rows:
            series.setdefault((row["node_id"], row["feature_name"]), []).append(row)
        for values in series.values():
            values.sort(key=lambda row: row["timestamp"])
        return series

    def _load_state_series(self) -> dict[str, list[dict[str, Any]]]:
        rows = self._load_jsonl("node_state_series.jsonl")
        series: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            series.setdefault(row["node_id"], []).append(row)
        for values in series.values():
            values.sort(key=lambda row: row["timestamp"])
        return series
