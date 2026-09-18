from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

from bs4 import BeautifulSoup
import pymupdf
import requests

from .schema import EvidenceRecord, Stage0ValidationError


DECISION_SOURCE_REGISTRY_VERSION = "stage3-decision-source-registry-v3"
DECISION_SOURCE_SNAPSHOT_ID = "stage3-decision-sources-2024-12-31-v3"
DECISION_SOURCE_CUTOFF = "2024-12-31"
DECISION_SOURCE_MIN_DOCUMENTS_PER_PMT = 3
DECISION_SOURCE_MIN_PUBLISHERS_PER_PMT = 2
DECISION_SOURCE_SPEC_MANIFEST_SHA256 = "a5558376de013699dc20c5024df94f579b1876488f15dec03ad49f89fd8bdd54"
DECISION_SOURCE_RECORDS_SHA256 = "a38404457c6866ed9b6f9425db99585ad9b3ac6cd86169a261e814c6602f3b70"
DECISION_SOURCE_USER_AGENT = "CyberForesight-DecisionEvidenceBuilder/1.0"
DECISION_SOURCE_MAX_CHUNK_CHARS = 2400
DECISION_SOURCE_MIN_CHUNK_CHARS = 120


DECISION_GUIDANCE_CLAIM_TYPES: tuple[str, ...] = (
    "implementation_feasibility",
    "implementation_limitation",
    "reference_architecture",
    "implementation_how_to",
    "deployment_maturity",
    "operational_benefit",
    "adoption_benefit",
    "adoption_barrier",
    "resource_burden",
    "organizational_applicability",
    "cyber_governance",
    "governance_enablement",
    "compliance_constraint",
    "recommended_control",
    "prioritized_defensive_practice",
    "security_control",
)

DECISION_GUIDANCE_PROHIBITED_CLAIM_TYPES: tuple[str, ...] = (
    "exact_effectiveness",
    "universal_control_effectiveness",
    "product_roi",
    "product_tco",
    "exact_budget",
    "exact_staffing_count",
)


@dataclass(frozen=True)
class DecisionSourceSpec:
    source_id: str
    source_name: str
    source_document_id: str
    source_uri: str
    parser: str
    publication_date: str
    available_at: str
    pmt_ids: tuple[str, ...]
    version: str
    publisher_group: str = "NIST"
    allowed_claim_types: tuple[str, ...] = DECISION_GUIDANCE_CLAIM_TYPES
    prohibited_claim_types: tuple[str, ...] = DECISION_GUIDANCE_PROHIBITED_CLAIM_TYPES

    def to_dict(self) -> dict[str, Any]:
        output = asdict(self)
        for key in ("pmt_ids", "allowed_claim_types", "prohibited_claim_types"):
            output[key] = list(output[key])
        return output


DECISION_SOURCE_SPECS: tuple[DecisionSourceSpec, ...] = (
    # NLP / LLM: three independent NIST documents, all available before the fixed cutoff.
    DecisionSourceSpec(
        "DE01",
        "NIST Artificial Intelligence Risk Management Framework 1.0",
        "nist-ai-rmf-1-0",
        "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf",
        "pdf_chunks",
        "2023-01-26",
        "2023-01-26",
        ("PMT_NLP_LLM",),
        "NIST-AI-100-1",
    ),
    DecisionSourceSpec(
        "DE02",
        "NIST Artificial Intelligence Risk Management Framework: Generative AI Profile",
        "nist-ai-600-1",
        "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf",
        "pdf_chunks",
        "2024-07-26",
        "2024-07-26",
        ("PMT_NLP_LLM",),
        "NIST-AI-600-1",
    ),
    DecisionSourceSpec(
        "DE03",
        "NIST Secure Software Development Practices for Generative AI and Dual-Use Foundation Models",
        "nist-sp-800-218a",
        "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-218A.pdf",
        "pdf_chunks",
        "2024-07-26",
        "2024-07-26",
        ("PMT_NLP_LLM",),
        "NIST-SP-800-218A",
    ),
    # Anomaly Detection / IDS-IPS: implementation and operational deployment guidance.
    DecisionSourceSpec(
        "DE04",
        "NIST Guide to Intrusion Detection and Prevention Systems",
        "nist-sp-800-94",
        "https://nvlpubs.nist.gov/nistpubs/legacy/sp/nistspecialpublication800-94.pdf",
        "pdf_chunks",
        "2007-02-20",
        "2007-02-20",
        ("PMT_ANOMALY_DETECTION", "PMT_IDS_IPS"),
        "NIST-SP-800-94",
    ),
    DecisionSourceSpec(
        "DE05",
        "NIST Securing Manufacturing ICS: Behavioral Anomaly Detection",
        "nist-ir-8219",
        "https://nvlpubs.nist.gov/nistpubs/ir/2020/NIST.IR.8219.pdf",
        "pdf_chunks",
        "2020-07-16",
        "2020-07-16",
        ("PMT_ANOMALY_DETECTION",),
        "NIST-IR-8219",
    ),
    DecisionSourceSpec(
        "DE06",
        "NIST Protecting Information and System Integrity in Industrial Control System Environments",
        "nist-sp-1800-10",
        "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.1800-10.pdf",
        "pdf_chunks",
        "2022-03-16",
        "2022-03-16",
        ("PMT_ANOMALY_DETECTION", "PMT_IDS_IPS"),
        "NIST-SP-1800-10",
    ),
    DecisionSourceSpec(
        "DE07",
        "NIST Information Security Continuous Monitoring",
        "nist-sp-800-137",
        "https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-137.pdf",
        "pdf_chunks",
        "2011-09-30",
        "2011-09-30",
        ("PMT_IDS_IPS",),
        "NIST-SP-800-137",
    ),
    # Cryptography / HTTPS.
    DecisionSourceSpec(
        "DE08",
        "NIST Recommendation for Key Management Part 1 Rev. 5",
        "nist-sp-800-57pt1r5",
        "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-57pt1r5.pdf",
        "pdf_chunks",
        "2020-05-04",
        "2020-05-04",
        ("PMT_CRYPTOGRAPHY",),
        "NIST-SP-800-57PT1R5",
    ),
    DecisionSourceSpec(
        "DE09",
        "NIST Transitioning the Use of Cryptographic Algorithms and Key Lengths Rev. 2",
        "nist-sp-800-131ar2",
        "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-131Ar2.pdf",
        "pdf_chunks",
        "2019-03-21",
        "2019-03-21",
        ("PMT_CRYPTOGRAPHY",),
        "NIST-SP-800-131AR2",
    ),
    DecisionSourceSpec(
        "DE10",
        "NIST Guidelines for TLS Implementations Rev. 2",
        "nist-sp-800-52r2",
        "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-52r2.pdf",
        "pdf_chunks",
        "2019-08-29",
        "2019-08-29",
        ("PMT_CRYPTOGRAPHY", "PMT_HTTPS"),
        "NIST-SP-800-52R2",
    ),
    # Access control / identity architecture.  SP 800-63B also supplies a second HTTPS/TLS deployment source.
    DecisionSourceSpec(
        "DE11",
        "NIST Guide to Attribute Based Access Control",
        "nist-sp-800-162",
        "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-162.pdf",
        "pdf_chunks",
        "2014-01-31",
        "2014-01-31",
        ("PMT_ACCESS_CONTROL",),
        "NIST-SP-800-162",
    ),
    DecisionSourceSpec(
        "DE12",
        "NIST Zero Trust Architecture",
        "nist-sp-800-207",
        "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-207.pdf",
        "pdf_chunks",
        "2020-08-11",
        "2020-08-11",
        ("PMT_ACCESS_CONTROL",),
        "NIST-SP-800-207",
    ),
    DecisionSourceSpec(
        "DE13",
        "NIST Digital Identity Guidelines: Authentication and Lifecycle Management",
        "nist-sp-800-63b",
        "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-63b.pdf",
        "pdf_chunks",
        "2017-06-22",
        "2017-06-22",
        ("PMT_ACCESS_CONTROL", "PMT_HTTPS"),
        "NIST-SP-800-63B",
    ),
    DecisionSourceSpec(
        "DE14",
        "NIST Guide to SSL VPNs",
        "nist-sp-800-113",
        "https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-113.pdf",
        "pdf_chunks",
        "2008-07-01",
        "2008-07-01",
        ("PMT_HTTPS",),
        "NIST-SP-800-113",
    ),
    # Independent official sources are deliberately included so the main
    # decision-evidence universe is not a single-publisher NIST corpus.
    DecisionSourceSpec(
        "DE15",
        "UK NCSC Guidelines for Secure AI System Development",
        "ncsc-guidelines-secure-ai-system-development",
        "https://www.ncsc.gov.uk/sites/default/files/documents/Guidelines-for-secure-AI-system-development.pdf",
        "pdf_chunks",
        "2023-11-27",
        "2023-11-27",
        ("PMT_NLP_LLM",),
        "NCSC-SECURE-AI-1.0",
        publisher_group="NCSC_UK",
    ),
    DecisionSourceSpec(
        "DE16",
        "CISA/ICS-CERT Improving ICS Cybersecurity with Defense-in-Depth Strategies",
        "cisa-ics-defense-in-depth-2016",
        "https://www.cisa.gov/sites/default/files/2023-01/NCCIC_ICS-CERT_Defense_in_Depth_2016_S508C.pdf",
        "pdf_chunks",
        "2016-09-01",
        "2016-09-01",
        ("PMT_ANOMALY_DETECTION", "PMT_IDS_IPS"),
        "CISA-ICS-DID-2016",
        publisher_group="CISA_DHS",
    ),
    DecisionSourceSpec(
        "DE17",
        "CISA Zero Trust Maturity Model Version 2.0",
        "cisa-zero-trust-maturity-model-v2",
        "https://www.cisa.gov/sites/default/files/2023-04/zero_trust_maturity_model_v2_508.pdf",
        "pdf_chunks",
        "2023-04-01",
        "2023-04-01",
        ("PMT_ACCESS_CONTROL",),
        "CISA-ZTMM-2.0",
        publisher_group="CISA_DHS",
    ),
    DecisionSourceSpec(
        "DE18",
        "DHS Control Systems Communications Encryption Primer",
        "cisa-dhs-control-systems-encryption-primer",
        "https://www.cisa.gov/sites/default/files/documents/Encryption_Primer_20091211_S508C.pdf",
        "pdf_chunks",
        "2009-12-01",
        "2009-12-01",
        ("PMT_CRYPTOGRAPHY", "PMT_HTTPS"),
        "DHS-ENCRYPTION-PRIMER-2009",
        publisher_group="CISA_DHS",
    ),
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean_text(text: str) -> str:
    text = text.replace("\x00", " ").replace("\u00ad", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _chunks(text: str) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n{1,}", text) if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        pieces = (
            [paragraph[index : index + DECISION_SOURCE_MAX_CHUNK_CHARS] for index in range(0, len(paragraph), DECISION_SOURCE_MAX_CHUNK_CHARS)]
            if len(paragraph) > DECISION_SOURCE_MAX_CHUNK_CHARS
            else [paragraph]
        )
        for piece in pieces:
            if current and len(current) + len(piece) + 1 > DECISION_SOURCE_MAX_CHUNK_CHARS:
                if len(current) >= DECISION_SOURCE_MIN_CHUNK_CHARS:
                    chunks.append(current)
                current = piece
            else:
                current = f"{current}\n{piece}".strip()
    if len(current) >= DECISION_SOURCE_MIN_CHUNK_CHARS:
        chunks.append(current)
    return chunks


def decision_source_snapshot_dir(project_root: str | Path) -> Path:
    return (
        Path(project_root).resolve()
        / "Multi-Agent"
        / "Results"
        / "Stage0"
        / "DecisionEvidence"
        / "snapshots"
        / DECISION_SOURCE_SNAPSHOT_ID
    )


def _spec_manifest_sha256() -> str:
    return _sha256_bytes(_canonical_json([spec.to_dict() for spec in DECISION_SOURCE_SPECS]).encode("utf-8"))


def _make_record(spec: DecisionSourceSpec, *, locator: str, content: str) -> EvidenceRecord:
    normalized = _clean_text(content)
    content_hash = _sha256_bytes(normalized.encode("utf-8"))
    record = EvidenceRecord(
        evidence_id=f"DE_{spec.source_id}_{content_hash[:16].upper()}",
        source_registry_id=spec.source_id,
        source_family="J",
        source=spec.source_name,
        publication_date=spec.publication_date,
        evidence_date=None,
        available_at=spec.available_at,
        source_document_id=spec.source_document_id,
        source_locator=locator,
        source_snapshot_id=DECISION_SOURCE_SNAPSHOT_ID,
        evidence_chain_id=f"decision-guidance:{spec.source_document_id}",
        evidence_type="decision_guidance_chunk",
        retrieval_role="decision_guidance_curated",
        allowed_claim_types=spec.allowed_claim_types,
        prohibited_claim_types=spec.prohibited_claim_types,
        threat_ids=(),
        pmt_ids=tuple(sorted(spec.pmt_ids)),
        content=normalized,
        content_hash=content_hash,
    )
    record.validate()
    return record


def _parse_source(spec: DecisionSourceSpec, path: Path) -> list[EvidenceRecord]:
    records: list[EvidenceRecord] = []
    if spec.parser == "pdf_chunks":
        with pymupdf.open(path) as document:
            for page_number, page in enumerate(document, start=1):
                text = _clean_text(page.get_text("text") or "")
                for chunk_number, chunk in enumerate(_chunks(text), start=1):
                    records.append(
                        _make_record(
                            spec,
                            locator=f"page:{page_number}:chunk:{chunk_number}",
                            content=chunk,
                        )
                    )
    elif spec.parser == "html_chunks":
        soup = BeautifulSoup(path.read_bytes(), "html.parser")
        for node in soup(["script", "style", "noscript", "svg"]):
            node.decompose()
        target = soup.find("main") or soup.find("article") or soup.body or soup
        text = _clean_text(target.get_text("\n", strip=True))
        for chunk_number, chunk in enumerate(_chunks(text), start=1):
            records.append(
                _make_record(
                    spec,
                    locator=f"{spec.source_uri}#chunk-{chunk_number}",
                    content=chunk,
                )
            )
    else:
        raise Stage0ValidationError(f"Unsupported decision-source parser {spec.parser!r}")
    if not records:
        raise Stage0ValidationError(f"Decision source {spec.source_document_id} produced zero records")
    return records


def _source_coverage(specs: tuple[DecisionSourceSpec, ...] = DECISION_SOURCE_SPECS) -> dict[str, list[str]]:
    coverage: dict[str, set[str]] = {}
    for spec in specs:
        for pmt_id in spec.pmt_ids:
            coverage.setdefault(pmt_id, set()).add(spec.source_document_id)
    return {pmt_id: sorted(documents) for pmt_id, documents in sorted(coverage.items())}


def _publisher_coverage(
    specs: tuple[DecisionSourceSpec, ...] = DECISION_SOURCE_SPECS,
) -> dict[str, list[str]]:
    coverage: dict[str, set[str]] = {}
    for spec in specs:
        for pmt_id in spec.pmt_ids:
            coverage.setdefault(pmt_id, set()).add(spec.publisher_group)
    return {pmt_id: sorted(publishers) for pmt_id, publishers in sorted(coverage.items())}


def validate_decision_source_spec_coverage() -> dict[str, Any]:
    cutoff = date.fromisoformat(DECISION_SOURCE_CUTOFF)
    coverage = _source_coverage()
    publisher_coverage = _publisher_coverage()
    required_pmts = {
        "PMT_NLP_LLM",
        "PMT_ANOMALY_DETECTION",
        "PMT_CRYPTOGRAPHY",
        "PMT_IDS_IPS",
        "PMT_ACCESS_CONTROL",
        "PMT_HTTPS",
    }
    issues: list[str] = []
    ids = [spec.source_id for spec in DECISION_SOURCE_SPECS]
    documents = [spec.source_document_id for spec in DECISION_SOURCE_SPECS]
    if len(ids) != len(set(ids)):
        issues.append("decision source IDs must be unique")
    if len(documents) != len(set(documents)):
        issues.append("decision source document IDs must be unique")
    for spec in DECISION_SOURCE_SPECS:
        if date.fromisoformat(spec.available_at) > cutoff:
            issues.append(f"{spec.source_document_id} is post-cutoff ({spec.available_at})")
        if not spec.pmt_ids:
            issues.append(f"{spec.source_document_id} has no curated PMT mapping")
    for pmt_id in sorted(required_pmts):
        document_count = len(coverage.get(pmt_id, []))
        if document_count < DECISION_SOURCE_MIN_DOCUMENTS_PER_PMT:
            issues.append(
                f"{pmt_id} has only {document_count} decision guidance documents; "
                f"minimum is {DECISION_SOURCE_MIN_DOCUMENTS_PER_PMT}"
            )
        publisher_count = len(publisher_coverage.get(pmt_id, []))
        if publisher_count < DECISION_SOURCE_MIN_PUBLISHERS_PER_PMT:
            issues.append(
                f"{pmt_id} has only {publisher_count} decision-guidance publishers; "
                f"minimum is {DECISION_SOURCE_MIN_PUBLISHERS_PER_PMT}"
            )
    actual_spec_hash = _spec_manifest_sha256()
    if actual_spec_hash != DECISION_SOURCE_SPEC_MANIFEST_SHA256:
        issues.append(
            "decision-source specification changed without a new version/hash: "
            f"{actual_spec_hash} != {DECISION_SOURCE_SPEC_MANIFEST_SHA256}"
        )
    if issues:
        raise Stage0ValidationError("Decision-source specification coverage failed: " + "; ".join(issues))
    return {
        "registry_version": DECISION_SOURCE_REGISTRY_VERSION,
        "snapshot_id": DECISION_SOURCE_SNAPSHOT_ID,
        "cutoff": DECISION_SOURCE_CUTOFF,
        "minimum_documents_per_pmt": DECISION_SOURCE_MIN_DOCUMENTS_PER_PMT,
        "minimum_publishers_per_pmt": DECISION_SOURCE_MIN_PUBLISHERS_PER_PMT,
        "document_coverage_by_pmt": coverage,
        "publisher_coverage_by_pmt": publisher_coverage,
        "manifest_sha256": DECISION_SOURCE_SPEC_MANIFEST_SHA256,
    }


def build_decision_source_snapshot(
    *,
    project_root: str | Path,
    timeout_seconds: int = 45,
) -> dict[str, Any]:
    """Build the immutable Stage-4 decision-guidance supplement.

    This deliberately does not mutate or regenerate the Stage-0 v4 evidence snapshot.
    It is a separate, cutoff-frozen supplement consumed only by contextual Stage 3.
    """

    validation = validate_decision_source_spec_coverage()
    snapshot_dir = decision_source_snapshot_dir(project_root)
    if snapshot_dir.exists():
        raise Stage0ValidationError(f"Decision-source snapshot already exists and is immutable: {snapshot_dir}")

    staging = Path(tempfile.mkdtemp(prefix="decision-source-ingest-"))
    artifacts_dir = snapshot_dir / "artifacts"
    session = requests.Session()
    session.headers.update({"User-Agent": DECISION_SOURCE_USER_AGENT, "Accept-Encoding": "gzip, deflate"})
    artifacts: list[dict[str, Any]] = []
    records_by_id: dict[str, EvidenceRecord] = {}
    try:
        artifacts_dir.mkdir(parents=True, exist_ok=False)
        for index, spec in enumerate(DECISION_SOURCE_SPECS, start=1):
            response = session.get(spec.source_uri, timeout=timeout_seconds, allow_redirects=True)
            response.raise_for_status()
            suffix = ".pdf" if spec.parser == "pdf_chunks" else ".html"
            staged = staging / f"{spec.source_id}_{spec.source_document_id}{suffix}"
            staged.write_bytes(response.content)
            if staged.stat().st_size == 0:
                raise Stage0ValidationError(f"Downloaded empty decision source {spec.source_document_id}")
            digest = _sha256_file(staged)
            destination = artifacts_dir / f"{spec.source_id}_{spec.source_document_id}_{digest[:12]}{suffix}"
            shutil.copy2(staged, destination)
            parsed = _parse_source(spec, destination)
            for record in parsed:
                if record.evidence_id in records_by_id:
                    continue
                records_by_id[record.evidence_id] = record
            artifacts.append(
                {
                    "source_id": spec.source_id,
                    "source_name": spec.source_name,
                    "source_document_id": spec.source_document_id,
                    "source_uri": spec.source_uri,
                    "resolved_uri": response.url,
                    "publication_date": spec.publication_date,
                    "available_at": spec.available_at,
                    "version": spec.version,
                    "publisher_group": spec.publisher_group,
                    "pmt_ids": list(spec.pmt_ids),
                    "content_hash": digest,
                    "snapshot_path": str(destination.relative_to(snapshot_dir)).replace("\\", "/"),
                    "record_count": len(parsed),
                }
            )
            print(
                f"[decision-source {index}/{len(DECISION_SOURCE_SPECS)}] "
                f"{spec.source_document_id}: {len(parsed)} records"
            )

        records = sorted(records_by_id.values(), key=lambda item: item.evidence_id)
        records_path = snapshot_dir / "records.jsonl"
        with records_path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
        records_sha256 = _sha256_file(records_path)
        if records_sha256 != DECISION_SOURCE_RECORDS_SHA256:
            raise Stage0ValidationError(
                "Decision-source records changed for the frozen v2 contract: "
                f"{records_sha256} != {DECISION_SOURCE_RECORDS_SHA256}"
            )
        manifest = {
            **validation,
            "immutable": True,
            "artifact_count": len(artifacts),
            "record_count": len(records),
            "artifacts": artifacts,
            "records_sha256": DECISION_SOURCE_RECORDS_SHA256,
            "specs": [spec.to_dict() for spec in DECISION_SOURCE_SPECS],
        }
        manifest_path = snapshot_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return manifest
    except BaseException:
        shutil.rmtree(snapshot_dir, ignore_errors=True)
        raise
    finally:
        session.close()
        shutil.rmtree(staging, ignore_errors=True)


def load_decision_source_snapshot(*, project_root: str | Path) -> tuple[list[EvidenceRecord], dict[str, Any]]:
    snapshot_dir = decision_source_snapshot_dir(project_root)
    manifest_path = snapshot_dir / "manifest.json"
    records_path = snapshot_dir / "records.jsonl"
    if not manifest_path.is_file() or not records_path.is_file():
        raise Stage0ValidationError(
            "Frozen Stage-4 decision-source snapshot is missing. "
            "Build it before contextual Stage 3 inference."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("immutable") is not True:
        raise Stage0ValidationError("Decision-source snapshot must declare immutable=true")
    if manifest.get("snapshot_id") != DECISION_SOURCE_SNAPSHOT_ID:
        raise Stage0ValidationError("Decision-source snapshot ID mismatch")
    if manifest.get("registry_version") != DECISION_SOURCE_REGISTRY_VERSION:
        raise Stage0ValidationError("Decision-source registry version mismatch")
    if manifest.get("manifest_sha256") != DECISION_SOURCE_SPEC_MANIFEST_SHA256:
        raise Stage0ValidationError("Decision-source specification manifest drift")
    if manifest.get("records_sha256") != DECISION_SOURCE_RECORDS_SHA256:
        raise Stage0ValidationError("Decision-source manifest records hash is not the frozen v2 hash")
    if _sha256_file(records_path) != DECISION_SOURCE_RECORDS_SHA256:
        raise Stage0ValidationError("Decision-source records hash mismatch")

    records: list[EvidenceRecord] = []
    ids: set[str] = set()
    with records_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = EvidenceRecord.from_dict(json.loads(line))
            record.validate()
            if record.evidence_id in ids:
                raise Stage0ValidationError(f"Duplicate decision-source evidence ID {record.evidence_id}")
            if record.source_snapshot_id != DECISION_SOURCE_SNAPSHOT_ID:
                raise Stage0ValidationError(f"Decision-source snapshot provenance mismatch for {record.evidence_id}")
            if date.fromisoformat(record.available_at) > date.fromisoformat(DECISION_SOURCE_CUTOFF):
                raise Stage0ValidationError(f"Post-cutoff decision evidence leaked: {record.evidence_id}")
            if record.retrieval_role != "decision_guidance_curated":
                raise Stage0ValidationError(f"Unexpected decision-source retrieval role: {record.evidence_id}")
            ids.add(record.evidence_id)
            records.append(record)
    if len(records) != int(manifest.get("record_count") or -1):
        raise Stage0ValidationError("Decision-source record count mismatch")
    validate_decision_source_spec_coverage()
    return records, manifest

