from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any


CONTEXT_SCENARIO_SET_VERSION = "stage3-context-scenarios-v1"
CONTEXT_REFERENCE_PROFILE_VERSION = "cis-ig2-reference-v1"
CONTEXT_SELECTION_RULE_VERSION = "cis-ig2-fixed-enterprise-jurisdiction-sweep-v1"


CIS_IG2_SOURCE = {
    "source_id": "CTX_CIS_IG2_V1",
    "source": "Center for Internet Security - CIS Controls v7.1, Implementation Group 2",
    "source_uri": "https://downloads.cisecurity.org/controls/CIS-Controls-Version-7-1.pdf",
    "source_content_sha256": "ae55cb37c6e5ed108601fc159e9c5e1e66d68d0f599499472cbddd14ed2fd984",
    "publication_date": "2019-03-30",
    "available_at": "2019-03-30",
    "source_locator": "page 9, Implementation Group 2",
    "content": (
        "CIS Implementation Group 2 describes an enterprise with personnel responsible for managing and protecting "
        "IT infrastructure across multiple departments with differing risk profiles. Such enterprises often process "
        "sensitive client or enterprise information, may face regulatory burdens, and can tolerate short service "
        "interruptions. IG2 safeguards address increased operational complexity; some depend on enterprise-grade "
        "technology and specialized expertise to install and configure."
    ),
    "methodological_reference": {
        "source": "CIS Guide to Implementation Groups (IG): CIS Critical Security Controls v8.1",
        "source_uri": (
            "https://www.cisecurity.org/insights/white-papers/"
            "guide-implementation-groups-ig-cis-critical-security-controls-v8-1"
        ),
        "publication_date": "2024-11-25",
        "role": (
            "Confirms that CIS Controls v8.1 continues to use IG1/IG2/IG3 and selects IGs using factors including "
            "size/complexity, data types, resources/technology, threat types, and risk. It is design provenance, "
            "not a substitute for the frozen IG2 profile evidence above."
        ),
    },
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _context_evidence_id() -> str:
    digest = hashlib.sha256(_canonical_json(CIS_IG2_SOURCE).encode("utf-8")).hexdigest()
    return f"CTX_CIS_IG2_{digest[:16].upper()}"


def cis_ig2_context_evidence_record() -> dict[str, Any]:
    content = CIS_IG2_SOURCE["content"]
    return {
        "evidence_id": _context_evidence_id(),
        "source_registry_id": "CTX_CIS_IG2",
        "source_family": "CONTEXT_STANDARD",
        "source": CIS_IG2_SOURCE["source"],
        "publication_date": CIS_IG2_SOURCE["publication_date"],
        "evidence_date": CIS_IG2_SOURCE["publication_date"],
        "available_at": CIS_IG2_SOURCE["available_at"],
        "source_document_id": CONTEXT_REFERENCE_PROFILE_VERSION,
        "source_locator": CIS_IG2_SOURCE["source_locator"],
        "source_snapshot_id": CONTEXT_SCENARIO_SET_VERSION,
        "evidence_chain_id": "CIS_CONTROLS_V7_1_IG2",
        "evidence_type": "context_standard_profile",
        "retrieval_role": "context_definition",
        "allowed_claim_types": [
            "organizational_context",
            "resource_profile",
            "infrastructure_context",
            "adoption_context",
        ],
        "prohibited_claim_types": [
            "exact_budget",
            "exact_staffing_count",
            "product_roi",
            "product_tco",
            "universal_control_effectiveness",
        ],
        "threat_ids": [],
        "pmt_ids": [],
        "content": content,
        "content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "source_uri": CIS_IG2_SOURCE["source_uri"],
        "source_content_sha256": CIS_IG2_SOURCE["source_content_sha256"],
    }


_BASE_IG2_CONTEXT = {
    "organization_type": "CIS_IG2_REFERENCE_ENTERPRISE",
    "infrastructure_context": (
        "Multi-department enterprise IT with personnel responsible for managing and protecting infrastructure; "
        "the environment may require enterprise-grade technology and specialized expertise for some safeguards."
    ),
    "budget_context": (
        "No numeric budget, price, ROI, TCO, staffing count, or procurement ceiling is assumed. Resource capacity is "
        "represented only by the frozen CIS IG2 reference profile; exact financial quantities remain unknown."
    ),
    "reference_enterprise_profile": "CIS_IG2",
    "reference_profile_version": CONTEXT_REFERENCE_PROFILE_VERSION,
}


_JURISDICTIONS = (
    ("KR", "Republic of Korea", "30"),
    ("EU", "European Union", "31"),
    ("US", "United States", "32"),
)


def main_context_scenarios() -> list[dict[str, Any]]:
    evidence_id = _context_evidence_id()
    scenarios: list[dict[str, Any]] = []
    for code, region, legal_source_id in _JURISDICTIONS:
        scenarios.append(
            {
                "scenario_id": f"{code}__CIS_IG2",
                "scenario_set_version": CONTEXT_SCENARIO_SET_VERSION,
                "context_selection_rule": CONTEXT_SELECTION_RULE_VERSION,
                "jurisdiction_code": code,
                "jurisdiction_source_registry_id": legal_source_id,
                "region": region,
                **deepcopy(_BASE_IG2_CONTEXT),
                "context_evidence_ids": [evidence_id],
                "assumptions": [
                    "The reference enterprise profile is fixed before Stage 4 lens evaluation.",
                    "Jurisdiction is an experimental scenario parameter, not an observed property of a real organization.",
                    "The same CIS IG2 reference-enterprise profile is used in KR, EU, and US scenarios.",
                ],
                "non_assumptions": [
                    "No exact employee count or security-team size is assumed.",
                    "No exact budget, price, ROI, TCO, or payback period is assumed.",
                    "No vendor, product, product version, sector, or deployment architecture is assumed unless evidence supplies it.",
                ],
            }
        )
    return scenarios


def context_scenario_manifest() -> dict[str, Any]:
    manifest = {
        "scenario_set_version": CONTEXT_SCENARIO_SET_VERSION,
        "reference_profile_version": CONTEXT_REFERENCE_PROFILE_VERSION,
        "context_selection_rule": CONTEXT_SELECTION_RULE_VERSION,
        "context_source": deepcopy(CIS_IG2_SOURCE),
        "context_evidence_record": cis_ig2_context_evidence_record(),
        "scenarios": main_context_scenarios(),
        "main_estimand": (
            "Within each frozen threat-PMT case and identical CIS IG2 reference enterprise, vary only jurisdictional "
            "context (KR/EU/US) and jurisdiction-specific primary-law evidence before independent Stage 4 lens evaluation."
        ),
        "resource_profile_sensitivity": ["CIS_IG1", "CIS_IG3"],
    }
    manifest["manifest_sha256"] = hashlib.sha256(_canonical_json(manifest).encode("utf-8")).hexdigest()
    return manifest


def ensure_frozen_context_scenario_manifest(*, project_root: str | Path) -> Path:
    """Freeze the contextual experiment definition before any lens inference.

    Reusing a scenario-set version with changed contents is rejected rather than
    silently creating a second experiment definition under the same label.
    """

    manifest = context_scenario_manifest()
    path = (
        Path(project_root).resolve()
        / "Multi-Agent"
        / "Results"
        / "Stage3"
        / "context_sets"
        / f"{CONTEXT_SCENARIO_SET_VERSION}.json"
    )
    serialized = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != serialized:
            raise RuntimeError(
                f"Frozen context scenario manifest changed in place: {path}. "
                "Create a new CONTEXT_SCENARIO_SET_VERSION instead of overwriting an experiment definition."
            )
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized, encoding="utf-8", newline="\n")
    return path

