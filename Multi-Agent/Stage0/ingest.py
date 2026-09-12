from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import re
import shutil
import tempfile
import time
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote, urlparse

import requests
import yaml
import pymupdf
from bs4 import BeautifulSoup

from .evidence_store import EvidenceSnapshotWriter, EvidenceStore
from .registry import SOURCE_REGISTRY_VERSION, registry_by_id, validate_source_registry
from .schema import Stage0ValidationError
from .source_manifest import ARTIFACT_SPECS, SOURCE_MANIFEST_VERSION, ArtifactSpec, required_external_source_ids


INGESTION_VERSION = "stage0-ingestion-v2"
USER_AGENT = "CyberForesight-Stage0-EvidenceBuilder/1.0"
MAX_CHUNK_CHARS = 2400
MIN_CHUNK_CHARS = 120
MAX_ACADEMIC_RESULTS_PER_QUERY = 5
MAX_EPSS_RECORDS_PER_SNAPSHOT = 5000


class EvidenceCorpusBuilder:
    """Create one immutable Stage 0 evidence snapshot from the frozen source manifest."""

    def __init__(
        self,
        project_root: str | Path,
        *,
        snapshot_date: str,
        snapshot_id: str | None = None,
        timeout_seconds: int = 60,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.snapshot_date = date.fromisoformat(snapshot_date).isoformat()
        self.snapshot_id = snapshot_id or f"stage0-{self.snapshot_date}"
        self.timeout_seconds = timeout_seconds
        self.forecast_dir = self.project_root / "Data" / "Forecast"
        self.evidence_root = self.project_root / "Data" / "Evidence"
        self.entities = self._load_json(self.forecast_dir / "node_registry.json")
        self.threat_entities = [e for e in self.entities if e["node_type"] == "threat"]
        self.pmt_entities = [e for e in self.entities if e["node_type"] == "pmt"]
        if len(self.threat_entities) != 26 or len(self.pmt_entities) != 98:
            raise Stage0ValidationError("Evidence ingestion requires canonical 26-threat / 98-PMT entity registry")
        self.aliases = self._build_alias_index()
        self.registry = registry_by_id()
        validate_source_registry()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"})
        self.writer: EvidenceSnapshotWriter | None = None
        self.acquisition_results: list[dict[str, Any]] = []
        self.parse_results: list[dict[str, Any]] = []

    def run(self) -> dict[str, Any]:
        self.writer = EvidenceSnapshotWriter(self.evidence_root, self.snapshot_id)
        staging = Path(tempfile.mkdtemp(prefix="stage0-ingest-"))
        try:
            for spec in ARTIFACT_SPECS:
                acquired = self._acquire(spec, staging)
                publication_date = spec.publication_date or self.snapshot_date
                available_at = spec.available_at or self.snapshot_date
                artifact = self.writer.add_artifact(
                    source_registry_id=spec.source_registry_id,
                    source_path=acquired,
                    artifact_id=spec.artifact_id,
                    publication_date=publication_date,
                    available_at=available_at,
                    version=spec.version if spec.version != "snapshot" else self.snapshot_date,
                    parser_version=f"{INGESTION_VERSION}:{spec.parser}",
                    source_uri=spec.locator,
                )
                snapshot_path = self.writer.snapshot_dir / artifact["snapshot_path"]
                before = len(self.writer.records)
                self._parse(spec, snapshot_path, publication_date, available_at)
                parsed = len(self.writer.records) - before
                self.parse_results.append(
                    {
                        "source_registry_id": spec.source_registry_id,
                        "artifact_id": spec.artifact_id,
                        "parser": spec.parser,
                        "record_count": parsed,
                    }
                )

            audit = self._coverage_audit()
            self.writer.set_audit_metadata(audit)
            snapshot_dir = self.writer.finalize()
            # Re-open through the read-only runtime to verify immutable integrity.
            store = EvidenceStore(snapshot_dir)
            if len(store.records) != audit["record_count"]:
                raise Stage0ValidationError("Finalized EvidenceStore record count differs from ingestion audit")
            return audit
        except BaseException:
            if self.writer is not None:
                self.writer.abort()
            raise
        finally:
            shutil.rmtree(staging, ignore_errors=True)
            self.session.close()

    # ------------------------------------------------------------------
    # Acquisition
    # ------------------------------------------------------------------
    def _acquire(self, spec: ArtifactSpec, staging: Path) -> Path:
        try:
            if spec.acquisition == "local":
                path = self.project_root / spec.locator
                if not path.is_file():
                    raise Stage0ValidationError(f"Required local source artifact is missing: {path}")
                self._record_acquisition(spec, "PASS", path.stat().st_size, None)
                return path
            if spec.acquisition == "url":
                self._validate_source_url(spec)
                path = staging / self._staging_name(spec)
                response = self._get_with_retry(spec.locator, allow_redirects=True)
                path.write_bytes(response.content)
                if path.stat().st_size == 0:
                    raise Stage0ValidationError(f"Downloaded empty artifact: {spec.locator}")
                self._record_acquisition(spec, "PASS", path.stat().st_size, response.url)
                return path
            if spec.acquisition == "arxiv_batch":
                path = staging / f"{spec.artifact_id}.xml"
                payload = self._acquire_arxiv()
                path.write_bytes(payload)
                self._record_acquisition(spec, "PASS", path.stat().st_size, spec.locator)
                return path
            if spec.acquisition == "crossref_batch":
                path = staging / f"{spec.artifact_id}.json"
                payload = self._acquire_crossref()
                path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
                self._record_acquisition(spec, "PASS", path.stat().st_size, spec.locator)
                return path
            raise Stage0ValidationError(f"Unknown acquisition mode {spec.acquisition!r}")
        except Exception as exc:
            self._record_acquisition(spec, "FAIL", 0, None, error=f"{type(exc).__name__}: {exc}")
            if spec.required:
                raise
            path = staging / f"optional-missing-{spec.artifact_id}.txt"
            path.write_text(f"Optional source unavailable: {exc}\n", encoding="utf-8")
            return path

    def _acquire_arxiv(self) -> bytes:
        # Keep arXiv complementary rather than using it as the sole PMT index.
        # Crossref below provides per-PMT publisher metadata; arXiv supplies an
        # independent open preprint view across the main technical subdomains.
        queries = (
            'all:"cybersecurity" AND (all:"mitigation" OR all:"defense")',
            'all:"network security" AND (all:"detection" OR all:"prevention")',
            'all:"adversarial machine learning"',
            'all:"deepfake detection"',
            'all:"hardware security"',
            'all:"formal verification" AND all:"security"',
            'all:"privacy preserving" AND all:"security"',
            'all:"zero trust" AND all:"security"',
        )
        feeds: list[bytes] = []
        for index, query in enumerate(queries):
            response = self._get_with_retry(
                "https://export.arxiv.org/api/query",
                params={"search_query": query, "start": 0, "max_results": 25, "sortBy": "submittedDate", "sortOrder": "descending"},
            )
            feeds.append(response.content)
            if index + 1 < len(queries):
                time.sleep(3.0)
        # One deterministic XML wrapper preserves each official Atom response
        # exactly as returned while allowing a single frozen artifact/hash.
        root = ET.Element("arxiv_snapshot", {"snapshot_date": self.snapshot_date})
        for query, feed in zip(queries, feeds):
            item = ET.SubElement(root, "query", {"search_query": query})
            atom = ET.fromstring(feed)
            item.append(atom)
        return ET.tostring(root, encoding="utf-8", xml_declaration=True)

    def _acquire_crossref(self) -> dict[str, Any]:
        queries: list[dict[str, Any]] = []
        for index, entity in enumerate(self.pmt_entities, start=1):
            query = entity["canonical_name"]
            params = {
                "query.bibliographic": query,
                "filter": f"from-pub-date:2011-07-01,until-pub-date:{self.snapshot_date}",
                "rows": MAX_ACADEMIC_RESULTS_PER_QUERY,
            }
            response = self._get_with_retry("https://api.crossref.org/works", params=params)
            message = response.json().get("message", {})
            queries.append(
                {
                    "pmt_id": entity["node_id"],
                    "query": query,
                    "results": message.get("items", []),
                    "total_results": message.get("total-results"),
                }
            )
            if index % 20 == 0:
                time.sleep(0.25)
        return {
            "source": "Crossref",
            "snapshot_date": self.snapshot_date,
            "query_count": len(queries),
            "queries": queries,
        }

    def _get_with_retry(self, url: str, *, params: dict[str, Any] | None = None, allow_redirects: bool = True) -> requests.Response:
        last_error: Exception | None = None
        for attempt in range(5):
            try:
                response = self.session.get(
                    url,
                    params=params,
                    timeout=self.timeout_seconds,
                    allow_redirects=allow_redirects,
                )
                if response.status_code not in {429, 500, 502, 503, 504}:
                    response.raise_for_status()
                    return response
                last_error = requests.HTTPError(f"HTTP {response.status_code} for {response.url}", response=response)
                retry_after = response.headers.get("Retry-After")
                delay = min(float(retry_after), 30.0) if retry_after and retry_after.isdigit() else min(2.0 ** attempt, 16.0)
                time.sleep(delay)
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_error = exc
                time.sleep(min(2.0 ** attempt, 16.0))
        raise Stage0ValidationError(f"HTTP acquisition failed after retries: {url}: {last_error}")

    def _validate_source_url(self, spec: ArtifactSpec) -> None:
        definition = self.registry[spec.source_registry_id]
        host = (urlparse(spec.locator).hostname or "").lower()
        allowed = tuple(domain.lower() for domain in definition.official_domains)
        if not allowed:
            return
        if not any(host == domain or host.endswith("." + domain) for domain in allowed):
            raise Stage0ValidationError(
                f"Source {spec.source_registry_id} URL host {host!r} is outside fixed official domains {allowed}"
            )

    @staticmethod
    def _staging_name(spec: ArtifactSpec) -> str:
        suffix = Path(urlparse(spec.locator).path).suffix.lower()
        if not suffix or len(suffix) > 8:
            suffix = {
                "pdf_chunks": ".pdf",
                "epss_csv_gz": ".csv.gz",
                "attack_stix": ".json",
                "nvd_json": ".json",
                "kev_json": ".json",
                "d3fend_jsonld": ".json",
                "atlas_yaml": ".yaml",
            }.get(spec.parser, ".html")
        return f"{spec.source_registry_id}_{spec.artifact_id}{suffix}"

    def _record_acquisition(
        self,
        spec: ArtifactSpec,
        status: str,
        byte_count: int,
        resolved_uri: str | None,
        *,
        error: str | None = None,
    ) -> None:
        self.acquisition_results.append(
            {
                "source_registry_id": spec.source_registry_id,
                "artifact_id": spec.artifact_id,
                "status": status,
                "bytes": byte_count,
                "configured_uri": spec.locator,
                "resolved_uri": resolved_uri,
                "error": error,
            }
        )

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------
    def _parse(self, spec: ArtifactSpec, path: Path, publication_date: str, available_at: str) -> None:
        parser = {
            "pdf_chunks": self._parse_pdf,
            "html_chunks": self._parse_html,
            "text_chunks": self._parse_text,
            "json_chunks": self._parse_json_chunks,
            "cisa_csaf_zip": self._parse_cisa_csaf_zip,
            "nvd_json": self._parse_nvd,
            "kev_json": self._parse_kev,
            "epss_csv_gz": self._parse_epss,
            "attack_stix": self._parse_attack,
            "d3fend_jsonld": self._parse_d3fend,
            "atlas_yaml": self._parse_atlas,
            "arxiv_atom": self._parse_arxiv,
            "crossref_json": self._parse_crossref,
        }.get(spec.parser)
        if parser is None:
            raise Stage0ValidationError(f"No parser implemented for {spec.parser!r}")
        parser(spec, path, publication_date, available_at)

    def _parse_text(self, spec: ArtifactSpec, path: Path, publication_date: str, available_at: str) -> None:
        text = self._clean_text(path.read_text(encoding="utf-8", errors="replace"))
        chunks = self._chunks(text)
        if not chunks:
            raise Stage0ValidationError(f"Text source {spec.artifact_id} produced no chunks")
        for chunk_no, chunk in enumerate(chunks, start=1):
            threats, pmts = self._tag_entities(chunk)
            self._add_record(
                spec,
                publication_date,
                available_at,
                source_locator=f"{spec.locator}#chunk-{chunk_no}",
                content=chunk,
                threat_ids=threats,
                pmt_ids=pmts,
            )

    def _parse_json_chunks(self, spec: ArtifactSpec, path: Path, publication_date: str, available_at: str) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        text = self._clean_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        chunks = self._chunks(text)
        if not chunks:
            raise Stage0ValidationError(f"JSON source {spec.artifact_id} produced no text chunks")
        for chunk_no, chunk in enumerate(chunks, start=1):
            threats, pmts = self._tag_entities(chunk)
            self._add_record(
                spec,
                publication_date,
                available_at,
                source_locator=f"{spec.locator}#json-chunk-{chunk_no}",
                content=chunk,
                threat_ids=threats,
                pmt_ids=pmts,
            )

    def _parse_cisa_csaf_zip(self, spec: ArtifactSpec, path: Path, publication_date: str, available_at: str) -> None:
        count = 0
        with zipfile.ZipFile(path) as archive:
            names = [
                name
                for name in archive.namelist()
                if "/csaf_files/" in name and name.lower().endswith(".json")
            ]
            if not names:
                raise Stage0ValidationError("CISA CSAF archive contains no advisory JSON files")
            for name in names:
                try:
                    payload = json.loads(archive.read(name).decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                document = payload.get("document", {})
                tracking = document.get("tracking", {})
                advisory_id = str(tracking.get("id") or Path(name).stem)
                initial = self._date_part(tracking.get("initial_release_date")) or publication_date
                current = self._date_part(tracking.get("current_release_date")) or available_at
                notes = document.get("notes", [])
                note_text = " | ".join(
                    self._clean_text(str(note.get("text") or ""))
                    for note in notes
                    if isinstance(note, dict) and note.get("text")
                )
                vulns = payload.get("vulnerabilities", [])
                cves = [
                    str(v.get("cve"))
                    for v in vulns
                    if isinstance(v, dict) and v.get("cve")
                ]
                products: list[str] = []
                product_tree = payload.get("product_tree", {})
                for branch in product_tree.get("branches", []) if isinstance(product_tree, dict) else []:
                    if isinstance(branch, dict):
                        products.extend(self._collect_csaf_product_names(branch))
                content = self._clean_text(
                    f"{advisory_id} | CVEs={','.join(cves)} | products={'; '.join(products[:40])} | {note_text}"
                )
                if len(content) < MIN_CHUNK_CHARS:
                    continue
                threats, pmts = self._tag_entities(content)
                self._add_record(
                    spec,
                    initial,
                    current,
                    evidence_date=initial,
                    evidence_chain_id=advisory_id,
                    source_locator=f"CISA-CSAF:{advisory_id}",
                    content=content,
                    threat_ids=threats,
                    pmt_ids=pmts,
                )
                count += 1
        if count == 0:
            raise Stage0ValidationError("CISA CSAF parser produced zero evidence records")

    def _collect_csaf_product_names(self, node: dict[str, Any]) -> list[str]:
        names: list[str] = []
        name = node.get("name")
        if name:
            names.append(str(name))
        product = node.get("product")
        if isinstance(product, dict) and product.get("name"):
            names.append(str(product["name"]))
        for child in node.get("branches", []) if isinstance(node.get("branches"), list) else []:
            if isinstance(child, dict):
                names.extend(self._collect_csaf_product_names(child))
        return names

    def _parse_pdf(self, spec: ArtifactSpec, path: Path, publication_date: str, available_at: str) -> None:
        with pymupdf.open(path) as document:
            for page_no, page in enumerate(document, start=1):
                text = self._clean_text(page.get_text("text") or "")
                for chunk_no, chunk in enumerate(self._chunks(text), start=1):
                    threats, pmts = self._tag_entities(chunk)
                    self._add_record(
                        spec,
                        publication_date,
                        available_at,
                        source_locator=f"page:{page_no}:chunk:{chunk_no}",
                        content=chunk,
                        threat_ids=threats,
                        pmt_ids=pmts,
                    )

    def _parse_html(self, spec: ArtifactSpec, path: Path, publication_date: str, available_at: str) -> None:
        raw = path.read_bytes()
        soup = BeautifulSoup(raw, "html.parser")
        for node in soup(["script", "style", "noscript", "svg"]):
            node.decompose()
        target = soup.find("main") or soup.find("article") or soup.body or soup
        text = self._clean_text(target.get_text("\n", strip=True))
        for chunk_no, chunk in enumerate(self._chunks(text), start=1):
            threats, pmts = self._tag_entities(chunk)
            self._add_record(
                spec,
                publication_date,
                available_at,
                source_locator=f"{spec.locator}#chunk-{chunk_no}",
                content=chunk,
                threat_ids=threats,
                pmt_ids=pmts,
            )

    def _parse_kev(self, spec: ArtifactSpec, path: Path, publication_date: str, available_at: str) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        vulnerabilities = payload.get("vulnerabilities", [])
        if not vulnerabilities:
            raise Stage0ValidationError("CISA KEV artifact contains no vulnerabilities")
        for item in vulnerabilities:
            cve = item.get("cveID", "").strip()
            if not cve:
                continue
            added = item.get("dateAdded") or available_at
            content = self._clean_text(
                " | ".join(
                    str(v)
                    for v in (
                        cve,
                        item.get("vendorProject"),
                        item.get("product"),
                        item.get("vulnerabilityName"),
                        item.get("shortDescription"),
                        item.get("requiredAction"),
                        item.get("knownRansomwareCampaignUse"),
                    )
                    if v
                )
            )
            threats, pmts = self._tag_entities(content)
            self._add_record(
                spec,
                added,
                added,
                evidence_date=added,
                evidence_chain_id=cve,
                source_locator=f"KEV:{cve}",
                content=content,
                threat_ids=threats,
                pmt_ids=pmts,
            )

    def _parse_nvd(self, spec: ArtifactSpec, path: Path, publication_date: str, available_at: str) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        vulnerabilities = payload.get("vulnerabilities", [])
        if not vulnerabilities:
            raise Stage0ValidationError("NVD hasKev query returned no vulnerabilities")
        for wrapper in vulnerabilities:
            cve = wrapper.get("cve", {})
            cve_id = cve.get("id", "").strip()
            if not cve_id:
                continue
            descriptions = cve.get("descriptions", [])
            english = next((d.get("value", "") for d in descriptions if d.get("lang") == "en"), "")
            metrics = cve.get("metrics", {})
            weaknesses = cve.get("weaknesses", [])
            content = self._clean_text(
                f"{cve_id} | {english} | metrics={json.dumps(metrics, ensure_ascii=False)[:1800]} | "
                f"weaknesses={json.dumps(weaknesses, ensure_ascii=False)[:900]}"
            )
            published = self._date_part(cve.get("published")) or publication_date
            modified = self._date_part(cve.get("lastModified")) or available_at
            threats, pmts = self._tag_entities(content)
            self._add_record(
                spec,
                published,
                modified,
                evidence_chain_id=cve_id,
                source_locator=f"NVD:{cve_id}",
                content=content,
                threat_ids=threats,
                pmt_ids=pmts,
            )

    def _parse_epss(self, spec: ArtifactSpec, path: Path, publication_date: str, available_at: str) -> None:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            first = f.readline().strip()
            score_date_match = re.search(r"score_date=([0-9-]{10})", first)
            score_date = score_date_match.group(1) if score_date_match else available_at
            rows = list(csv.DictReader(f))
        scored: list[tuple[float, dict[str, str]]] = []
        for row in rows:
            try:
                scored.append((float(row.get("epss", "0")), row))
            except ValueError:
                continue
        scored.sort(key=lambda pair: pair[0], reverse=True)
        for score, row in scored[:MAX_EPSS_RECORDS_PER_SNAPSHOT]:
            cve = row.get("cve", "").strip()
            if not cve:
                continue
            percentile = row.get("percentile", "")
            content = f"{cve} | EPSS={score:.8f} | percentile={percentile} | score_date={score_date}"
            self._add_record(
                spec,
                score_date,
                score_date,
                evidence_date=score_date,
                evidence_chain_id=cve,
                source_locator=f"EPSS:{score_date}:{cve}",
                content=content,
            )

    def _parse_attack(self, spec: ArtifactSpec, path: Path, publication_date: str, available_at: str) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        objects = payload.get("objects", [])
        for obj in objects:
            if obj.get("type") not in {"attack-pattern", "malware", "intrusion-set", "course-of-action"}:
                continue
            name = obj.get("name", "")
            description = obj.get("description", "")
            if not name:
                continue
            ext_id = ""
            for ref in obj.get("external_references", []):
                if ref.get("source_name") == "mitre-attack":
                    ext_id = ref.get("external_id", "")
                    break
            content = self._clean_text(f"{ext_id} | {name} | {description}")
            modified = self._date_part(obj.get("modified")) or available_at
            created = self._date_part(obj.get("created")) or publication_date
            threats, pmts = self._tag_entities(content)
            self._add_record(
                spec,
                created,
                modified,
                source_locator=f"ATT&CK:{ext_id or obj.get('id', name)}",
                content=content,
                threat_ids=threats,
                pmt_ids=pmts,
            )

    def _parse_d3fend(self, spec: ArtifactSpec, path: Path, publication_date: str, available_at: str) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        graph = payload.get("@graph", [])
        if not isinstance(graph, list) or not graph:
            raise Stage0ValidationError("D3FEND JSON-LD has no @graph records")
        for obj in graph:
            if not isinstance(obj, dict):
                continue
            label = self._jsonld_text(obj, "label")
            definition = self._jsonld_text(obj, "definition") or self._jsonld_text(obj, "comment")
            identifier = str(obj.get("@id") or "")
            content = self._clean_text(" | ".join(v for v in (identifier, label, definition) if v))
            if len(content) < MIN_CHUNK_CHARS:
                continue
            threats, pmts = self._tag_entities(content)
            self._add_record(
                spec,
                publication_date,
                available_at,
                source_locator=f"D3FEND:{identifier or hashlib.sha1(content.encode()).hexdigest()[:12]}",
                content=content,
                threat_ids=threats,
                pmt_ids=pmts,
            )

    def _parse_atlas(self, spec: ArtifactSpec, path: Path, publication_date: str, available_at: str) -> None:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise Stage0ValidationError("ATLAS YAML root is not an object")
        groups = ("tactics", "techniques", "mitigations", "case-studies")
        count = 0
        for group in groups:
            raw = payload.get(group, {})
            items = raw.values() if isinstance(raw, dict) else raw if isinstance(raw, list) else []
            for obj in items:
                if not isinstance(obj, dict):
                    continue
                identifier = str(obj.get("id") or obj.get("external-id") or obj.get("name") or "")
                name = str(obj.get("name") or "")
                description = str(obj.get("description") or obj.get("summary") or "")
                content = self._clean_text(f"{identifier} | {name} | {description}")
                if len(content) < 40:
                    continue
                threats, pmts = self._tag_entities(content)
                self._add_record(
                    spec,
                    publication_date,
                    available_at,
                    source_locator=f"ATLAS:{group}:{identifier}",
                    content=content,
                    threat_ids=threats,
                    pmt_ids=pmts,
                )
                count += 1
        if count == 0:
            raise Stage0ValidationError("ATLAS parser produced zero records")

    def _parse_arxiv(self, spec: ArtifactSpec, path: Path, publication_date: str, available_at: str) -> None:
        root = ET.parse(path).getroot()
        atom_ns = "{http://www.w3.org/2005/Atom}"
        count = 0
        for query_node in root.findall("query"):
            feed = next(iter(query_node), None)
            if feed is None:
                continue
            for entry in feed.findall(f"{atom_ns}entry"):
                title = self._clean_text(entry.findtext(f"{atom_ns}title") or "")
                summary = self._clean_text(entry.findtext(f"{atom_ns}summary") or "")
                identifier = (entry.findtext(f"{atom_ns}id") or "").strip()
                published = (entry.findtext(f"{atom_ns}published") or "")[:10]
                if not title:
                    continue
                content = self._clean_text(f"{title} | {summary}")
                threats, pmts = self._tag_entities(content)
                self._add_record(
                    spec,
                    published or publication_date,
                    published or available_at,
                    source_locator=f"arXiv:{identifier or hashlib.sha1(title.encode()).hexdigest()[:12]}",
                    content=content,
                    threat_ids=threats,
                    pmt_ids=pmts,
                )
                count += 1
        if count == 0:
            raise Stage0ValidationError("arXiv parser produced zero records")

    def _parse_crossref(self, spec: ArtifactSpec, path: Path, publication_date: str, available_at: str) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for query in payload.get("queries", []):
            pmt_id = query["pmt_id"]
            for item in query.get("results", []):
                title_values = item.get("title") or []
                title = title_values[0] if title_values else ""
                if not title:
                    continue
                doi = item.get("DOI") or ""
                container = "; ".join(item.get("container-title") or [])
                abstract = BeautifulSoup(item.get("abstract") or "", "html.parser").get_text(" ", strip=True)
                pub = self._crossref_date(item) or publication_date
                content = self._clean_text(f"{title} | DOI={doi} | venue={container} | {abstract}")
                self._add_record(
                    spec,
                    pub,
                    pub,
                    source_locator=f"Crossref:{doi or hashlib.sha1(title.encode()).hexdigest()[:12]}",
                    content=content,
                    pmt_ids=(pmt_id,),
                )

    def _add_record(
        self,
        spec: ArtifactSpec,
        publication_date: str,
        available_at: str,
        *,
        source_locator: str,
        content: str,
        evidence_date: str | None = None,
        evidence_chain_id: str | None = None,
        threat_ids: Iterable[str] = (),
        pmt_ids: Iterable[str] = (),
    ) -> None:
        assert self.writer is not None
        # Avoid duplicate chunks from repeated navigation/footer text inside the same source.
        normalized = self._clean_text(content)
        if len(normalized) < 20:
            return
        try:
            self.writer.add_record(
                source_registry_id=spec.source_registry_id,
                source=self.registry[spec.source_registry_id].name,
                publication_date=publication_date,
                available_at=available_at,
                source_document_id=spec.artifact_id,
                source_locator=source_locator,
                evidence_type=spec.evidence_type,
                content=normalized,
                evidence_date=evidence_date,
                evidence_chain_id=evidence_chain_id,
                threat_ids=threat_ids,
                pmt_ids=pmt_ids,
            )
        except Stage0ValidationError as exc:
            if "Duplicate evidence content/ID" in str(exc):
                return
            raise

    # ------------------------------------------------------------------
    # Coverage / integrity audit
    # ------------------------------------------------------------------
    def _coverage_audit(self) -> dict[str, Any]:
        assert self.writer is not None
        artifacts_by_source: dict[str, int] = {}
        records_by_source: dict[str, int] = {}
        records_by_family: dict[str, int] = {}
        for artifact in self.writer.artifacts:
            sid = artifact["source_registry_id"]
            artifacts_by_source[sid] = artifacts_by_source.get(sid, 0) + 1
        for record in self.writer.records:
            records_by_source[record.source_registry_id] = records_by_source.get(record.source_registry_id, 0) + 1
            records_by_family[record.source_family] = records_by_family.get(record.source_family, 0) + 1

        missing_artifact_sources = [sid for sid in required_external_source_ids() if artifacts_by_source.get(sid, 0) == 0]
        missing_record_sources = [sid for sid in required_external_source_ids() if records_by_source.get(sid, 0) == 0]
        threat_coverage = {
            e["node_id"]: sum(e["node_id"] in r.threat_ids for r in self.writer.records) for e in self.threat_entities
        }
        pmt_coverage = {
            e["node_id"]: sum(e["node_id"] in r.pmt_ids for r in self.writer.records) for e in self.pmt_entities
        }
        uncovered_threats = sorted(entity_id for entity_id, count in threat_coverage.items() if count == 0)
        uncovered_pmts = sorted(entity_id for entity_id, count in pmt_coverage.items() if count == 0)
        failed_acquisitions = [r for r in self.acquisition_results if r["status"] != "PASS"]

        issues: list[str] = []
        if missing_artifact_sources:
            issues.append(f"missing required source artifacts: {missing_artifact_sources}")
        if missing_record_sources:
            issues.append(f"required sources produced zero evidence records: {missing_record_sources}")
        if failed_acquisitions:
            issues.append(f"source acquisition failures: {[r['artifact_id'] for r in failed_acquisitions]}")
        if uncovered_threats:
            issues.append(f"threats with zero tagged external evidence: {uncovered_threats}")
        if uncovered_pmts:
            issues.append(f"PMTs with zero tagged external evidence: {uncovered_pmts}")

        # Families B-I must all be populated. Family A is the canonical forecast and is audited separately.
        missing_families = [family for family in "BCDEFGHI" if records_by_family.get(family, 0) == 0]
        if missing_families:
            issues.append(f"empty evidence families: {missing_families}")

        audit = {
            "status": "PASS" if not issues else "FAIL",
            "source_registry_version": SOURCE_REGISTRY_VERSION,
            "source_manifest_version": SOURCE_MANIFEST_VERSION,
            "ingestion_version": INGESTION_VERSION,
            "snapshot_date": self.snapshot_date,
            "snapshot_id": self.snapshot_id,
            "artifact_count": len(self.writer.artifacts),
            "record_count": len(self.writer.records),
            "artifacts_by_source": dict(sorted(artifacts_by_source.items())),
            "records_by_source": dict(sorted(records_by_source.items())),
            "records_by_family": dict(sorted(records_by_family.items())),
            "threat_coverage": threat_coverage,
            "pmt_coverage": pmt_coverage,
            "uncovered_threats": uncovered_threats,
            "uncovered_pmts": uncovered_pmts,
            "acquisition": self.acquisition_results,
            "parsing": self.parse_results,
            "issues": issues,
        }
        if issues:
            raise Stage0ValidationError("Evidence corpus validation failed: " + "; ".join(issues))
        return audit

    # ------------------------------------------------------------------
    # Tagging / text helpers
    # ------------------------------------------------------------------
    def _build_alias_index(self) -> list[tuple[str, re.Pattern[str], str]]:
        curated: dict[str, tuple[str, ...]] = {
            "DDoS": ("ddos", "distributed denial of service", "distributed denial-of-service"),
            "MITM": ("mitm", "man in the middle", "man-in-the-middle"),
            "Advanced persistent threat": ("advanced persistent threat", "apt"),
            "IoT Device Attack": ("iot device", "iot attack", "internet of things"),
            "Data Poisoning": ("data poisoning", "poisoning attack"),
            "Disinformation/Misinformation": ("disinformation", "misinformation", "information manipulation"),
            "Brute Force Attack": ("brute force", "password guessing"),
            "Password Attack": ("password attack", "credential attack"),
            "Account Hijacking": ("account hijacking", "account takeover"),
            "Session Hijacking": ("session hijacking",),
            "DNS Spoofing": ("dns spoofing", "dns poisoning"),
            "Insider Threat": ("insider threat",),
            "Adversarial Attack": ("adversarial attack", "adversarial machine learning"),
            "Deepfake": ("deepfake", "synthetic media"),
            "Supply Chain": ("supply chain",),
            "Zero-day": ("zero-day", "zero day"),
            "Cryptojacking": ("cryptojacking", "crypto mining malware"),
        }
        result: list[tuple[str, re.Pattern[str], str]] = []
        for entity in self.entities:
            entity_id = entity["node_id"]
            name = entity["canonical_name"]
            aliases = {name, re.sub(r"\([^)]*\)", "", name).strip()}
            aliases.update(curated.get(name, ()))
            for alias in sorted(a for a in aliases if len(a.strip()) >= 3):
                normalized = alias.strip()
                pattern = re.compile(r"(?<![A-Za-z0-9])" + re.escape(normalized) + r"(?![A-Za-z0-9])", re.IGNORECASE)
                result.append((entity_id, pattern, entity["node_type"]))
        return result

    def _tag_entities(self, content: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
        threats: set[str] = set()
        pmts: set[str] = set()
        for entity_id, pattern, entity_type in self.aliases:
            if pattern.search(content):
                (threats if entity_type == "threat" else pmts).add(entity_id)
        return tuple(sorted(threats)), tuple(sorted(pmts))

    @staticmethod
    def _clean_text(text: str) -> str:
        text = text.replace("\x00", " ").replace("\u00ad", "")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n[ \t]+", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _chunks(text: str) -> list[str]:
        if not text:
            return []
        paragraphs = [p.strip() for p in re.split(r"\n{1,}", text) if p.strip()]
        chunks: list[str] = []
        current = ""
        for paragraph in paragraphs:
            if len(paragraph) > MAX_CHUNK_CHARS:
                pieces = [paragraph[i : i + MAX_CHUNK_CHARS] for i in range(0, len(paragraph), MAX_CHUNK_CHARS)]
            else:
                pieces = [paragraph]
            for piece in pieces:
                if current and len(current) + len(piece) + 1 > MAX_CHUNK_CHARS:
                    if len(current) >= MIN_CHUNK_CHARS:
                        chunks.append(current)
                    current = piece
                else:
                    current = f"{current}\n{piece}".strip()
        if len(current) >= MIN_CHUNK_CHARS:
            chunks.append(current)
        return chunks

    @staticmethod
    def _date_part(value: Any) -> str | None:
        if not value:
            return None
        match = re.match(r"(\d{4}-\d{2}-\d{2})", str(value))
        return match.group(1) if match else None

    @staticmethod
    def _jsonld_text(obj: dict[str, Any], suffix: str) -> str:
        for key, value in obj.items():
            if key.lower().endswith(suffix.lower()):
                if isinstance(value, str):
                    return value
                if isinstance(value, list):
                    parts: list[str] = []
                    for item in value:
                        if isinstance(item, str):
                            parts.append(item)
                        elif isinstance(item, dict):
                            parts.append(str(item.get("@value") or item.get("value") or ""))
                    return " ".join(p for p in parts if p)
                if isinstance(value, dict):
                    return str(value.get("@value") or value.get("value") or "")
        return ""

    @staticmethod
    def _crossref_date(item: dict[str, Any]) -> str | None:
        for key in ("published-print", "published-online", "published", "issued", "created"):
            value = item.get(key)
            if not isinstance(value, dict):
                continue
            parts = value.get("date-parts")
            if not parts or not parts[0]:
                continue
            year = int(parts[0][0])
            month = int(parts[0][1]) if len(parts[0]) > 1 else 12
            day = int(parts[0][2]) if len(parts[0]) > 2 else 31
            try:
                return date(year, month, day).isoformat()
            except ValueError:
                return date(year, month, 1).isoformat()
        return None

    @staticmethod
    def _load_json(path: Path) -> Any:
        if not path.is_file():
            raise Stage0ValidationError(f"Missing canonical prerequisite: {path}")
        return json.loads(path.read_text(encoding="utf-8"))


def build_evidence_snapshot(project_root: str | Path, snapshot_date: str, snapshot_id: str | None = None) -> dict[str, Any]:
    return EvidenceCorpusBuilder(project_root, snapshot_date=snapshot_date, snapshot_id=snapshot_id).run()

