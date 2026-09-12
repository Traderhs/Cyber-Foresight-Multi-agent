from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from enum import StrEnum
from typing import Any


class Stage0ValidationError(ValueError):
    """Raised when Stage 0 input/output violates a fixed contract."""


class EvaluationMode(StrEnum):
    EX_ANTE_REPLAY = "ex_ante_replay"
    CURRENT_REASSESSMENT = "current_reassessment"


class BasisType(StrEnum):
    FORECAST_ONLY = "FORECAST_ONLY"
    EXTERNAL_EVIDENCE = "EXTERNAL_EVIDENCE"
    FORECAST_PLUS_EVIDENCE = "FORECAST_PLUS_EVIDENCE"


class EvidenceSufficiency(StrEnum):
    SUFFICIENT = "SUFFICIENT"
    PARTIAL = "PARTIAL"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    INSUFFICIENT_CONTEXT = "INSUFFICIENT_CONTEXT"


class ClaimReferenceStatus(StrEnum):
    VALID = "VALID"
    UNSUPPORTED_CLAIM = "UNSUPPORTED_CLAIM"
    SOURCE_CONTRACT_VIOLATION = "SOURCE_CONTRACT_VIOLATION"
    TEMPORAL_VIOLATION = "TEMPORAL_VIOLATION"
    UNKNOWN_REFERENCE = "UNKNOWN_REFERENCE"


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    source_registry_id: str
    source_family: str
    source: str
    publication_date: str
    evidence_date: str | None
    available_at: str
    source_document_id: str
    source_locator: str
    source_snapshot_id: str
    evidence_chain_id: str | None
    evidence_type: str
    retrieval_role: str
    allowed_claim_types: tuple[str, ...]
    prohibited_claim_types: tuple[str, ...]
    threat_ids: tuple[str, ...]
    pmt_ids: tuple[str, ...]
    content: str
    content_hash: str

    def validate(self) -> None:
        required = {
            "evidence_id": self.evidence_id,
            "source_registry_id": self.source_registry_id,
            "source_family": self.source_family,
            "source": self.source,
            "publication_date": self.publication_date,
            "available_at": self.available_at,
            "source_document_id": self.source_document_id,
            "source_locator": self.source_locator,
            "source_snapshot_id": self.source_snapshot_id,
            "evidence_type": self.evidence_type,
            "content": self.content,
            "content_hash": self.content_hash,
        }
        missing = [k for k, v in required.items() if not v]
        if missing:
            raise Stage0ValidationError(f"EvidenceRecord missing required fields: {missing}")
        _parse_date(self.publication_date, "publication_date")
        _parse_date(self.available_at, "available_at")
        if self.evidence_date:
            _parse_date(self.evidence_date, "evidence_date")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for key in ("allowed_claim_types", "prohibited_claim_types", "threat_ids", "pmt_ids"):
            d[key] = list(d[key])
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EvidenceRecord":
        return cls(
            **{
                **d,
                "allowed_claim_types": tuple(d.get("allowed_claim_types", [])),
                "prohibited_claim_types": tuple(d.get("prohibited_claim_types", [])),
                "threat_ids": tuple(d.get("threat_ids", [])),
                "pmt_ids": tuple(d.get("pmt_ids", [])),
            }
        )


@dataclass(frozen=True)
class EvidencePack:
    case_id: str
    source_registry_version: str
    source_snapshot_id: str
    forecast_origin_date: str
    training_data_end: str
    analysis_cutoff_date: str
    evaluation_mode: EvaluationMode
    threat_id: str
    pmt_id: str
    forecast_summary: dict[str, Any]
    evidence: tuple[EvidenceRecord, ...]
    retrieval_metadata: dict[str, Any]

    def validate(self) -> None:
        cutoff = _parse_date(self.analysis_cutoff_date, "analysis_cutoff_date")
        origin = _parse_date(self.forecast_origin_date, "forecast_origin_date")
        training_end = _parse_date(self.training_data_end, "training_data_end")
        if training_end > origin:
            raise Stage0ValidationError(
                f"training_data_end {training_end.isoformat()} cannot be later than forecast_origin_date {origin.isoformat()}"
            )
        ids: set[str] = set()
        for record in self.evidence:
            record.validate()
            if record.evidence_id in ids:
                raise Stage0ValidationError(f"Duplicate evidence_id: {record.evidence_id}")
            ids.add(record.evidence_id)
            if _parse_date(record.available_at, "available_at") > cutoff:
                raise Stage0ValidationError(
                    f"Temporal violation: {record.evidence_id} available {record.available_at} after {cutoff.isoformat()}"
                )
        if self.retrieval_metadata.get("temporal_violation_count", 0) != 0:
            raise Stage0ValidationError("EvidencePack cannot contain temporal violations")

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "source_registry_version": self.source_registry_version,
            "source_snapshot_id": self.source_snapshot_id,
            "forecast_origin_date": self.forecast_origin_date,
            "training_data_end": self.training_data_end,
            "analysis_cutoff_date": self.analysis_cutoff_date,
            "evaluation_mode": self.evaluation_mode.value,
            "threat_id": self.threat_id,
            "pmt_id": self.pmt_id,
            "forecast_summary": self.forecast_summary,
            "evidence": [r.to_dict() for r in self.evidence],
            "retrieval_metadata": self.retrieval_metadata,
        }


@dataclass(frozen=True)
class ClaimReference:
    claim_id: str
    claim_type: str
    basis_type: BasisType
    evidence_ids: tuple[str, ...] = field(default_factory=tuple)
    forecast_field_ids: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ClaimReferenceAudit:
    claim_id: str
    status: ClaimReferenceStatus
    issues: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return self.status == ClaimReferenceStatus.VALID


def _parse_date(value: str, field_name: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise Stage0ValidationError(f"{field_name} must be ISO YYYY-MM-DD, got {value!r}") from exc
