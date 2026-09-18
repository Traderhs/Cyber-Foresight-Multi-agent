from __future__ import annotations

from datetime import date
from pathlib import Path
import re
from typing import Any

from .bm25 import BM25_B, BM25_K1, BM25_RETRIEVER_VERSION, rank_bm25
from .context_scenarios import cis_ig2_context_evidence_record
from .decision_sources import DECISION_SOURCE_SNAPSHOT_ID
from .evidence_store import EvidenceStore
from .schema import EvidenceRecord, Stage0ValidationError


DECISION_EVIDENCE_RETRIEVAL_VERSION = "decision-evidence-retrieval-v5"
MAX_RECORDS_PER_DECISION_SLOT = 3
MAIN_MIN_GENERIC_DECISION_EVIDENCE_RECORDS = 6
MAIN_MIN_GENERIC_DECISION_EVIDENCE_DOCUMENTS = 3
MAIN_MIN_LENS_DECISION_EVIDENCE_DOCUMENTS = 3
MAIN_MIN_BALANCE_DOCUMENTS_PER_FACET = 0
MAIN_MIN_BALANCE_PUBLISHERS_PER_FACET = 0
MAIN_MIN_LENS_DECISION_EVIDENCE_PUBLISHERS = 2
MAIN_MIN_ACTION_DIRECTION_RECORDS = 1
MAIN_LENS_DECISION_EVIDENCE_SLOTS: dict[str, frozenset[str]] = {
    "technical_feasibility": frozenset(
        {
            "technical_enablement",
            "technical_limitation",
            "deployment_maturity",
        }
    ),
    "institutional_regional": frozenset(
        {
            "governance_enablement",
            "compliance_constraint",
            "regulatory_applicability",
        }
    ),
    "financial_adoption": frozenset(
        {
            "adoption_benefit",
            "adoption_burden",
            "deployment_maturity",
        }
    ),
}

# v5 audits both directions but does not force equal counts. The retrieval
# procedure is symmetric; the observed evidence distribution may legitimately
# be asymmetric. This avoids manufacturing a 50:50 evidence universe.
MAIN_REQUIRED_BALANCE_FACETS_BY_LENS: dict[str, frozenset[str]] = {
    "technical_feasibility": frozenset(
        {"technical_enablement", "technical_limitation"}
    ),
    "institutional_regional": frozenset(
        {"governance_enablement", "compliance_constraint"}
    ),
    "financial_adoption": frozenset(
        {"adoption_benefit", "adoption_burden"}
    ),
}


_PMT_DECISION_ALIASES: dict[str, tuple[str, ...]] = {
    "PMT_NLP_LLM": (
        "nlp",
        "llm",
        "large language model",
        "large language models",
        "natural language processing",
    ),
    "PMT_ANOMALY_DETECTION": (
        "anomaly detection",
        "anomaly detector",
        "anomaly-based detection",
        "anomaly based detection",
    ),
    "PMT_CRYPTOGRAPHY": (
        "cryptography",
        "cryptographic",
        "encryption",
        "key management",
    ),
    "PMT_IDS_IPS": (
        "ids/ips",
        "intrusion detection system",
        "intrusion detection systems",
        "intrusion prevention system",
        "intrusion prevention systems",
        "intrusion detection",
        "intrusion prevention",
    ),
    "PMT_ACCESS_CONTROL": (
        "access control",
        "access controls",
        "identity and access management",
        "access management",
    ),
    "PMT_HTTPS": (
        "https",
        "transport layer security",
        "tls",
    ),
}


DECISION_EVIDENCE_SLOT_CLAIM_TYPES: dict[str, frozenset[str]] = {
    "technical_enablement": frozenset(
        {
            "action_effectiveness",
            "recommended_control",
            "security_control",
            "privacy_control",
            "implementation_feasibility",
            "reference_architecture",
            "implementation_how_to",
            "organizational_applicability",
            "operational_benefit",
        }
    ),
    "technical_limitation": frozenset(
        {
            "action_limitation",
            "implementation_limitation",
            "organizational_applicability",
            "cyber_governance",
            "implementation_feasibility",
            "reference_architecture",
            "implementation_how_to",
        }
    ),
    "deployment_maturity": frozenset(
        {
            "deployment_maturity",
            "organizational_applicability",
            "implementation_feasibility",
            "reference_architecture",
            "prioritized_defensive_practice",
            "recommended_control",
        }
    ),
    "adoption_benefit": frozenset(
        {
            "adoption_benefit",
            "operational_benefit",
        }
    ),
    "adoption_burden": frozenset(
        {
            "adoption_barrier",
            "resource_burden",
            "organizational_applicability",
            "cyber_governance",
            "implementation_feasibility",
            "implementation_limitation",
        }
    ),
    "regulatory_applicability": frozenset(
        {
            "legal_requirement",
            "regulatory_scope",
            "korea_control_requirement",
            "eu_legal_requirement",
            "us_legal_requirement",
        }
    ),
    "governance_enablement": frozenset(
        {
            "governance_enablement",
            "cyber_governance",
            "control_assessment_procedure",
            "implementation_assessment",
            "organizational_applicability",
        }
    ),
    "compliance_constraint": frozenset(
        {
            "compliance_constraint",
            "legal_requirement",
            "regulatory_scope",
            "cyber_governance",
            "organizational_applicability",
        }
    ),
}


_SLOT_QUERY_HINTS = {
    "technical_enablement": "technical feasibility enables supports implementation integration deployment automation interoperability benefit",
    "technical_limitation": "technical limitation challenge constraint complexity failure false positive false negative latency overhead drawback",
    "deployment_maturity": "deployment maturity production operational adoption implementation reference deployment real-world use",
    "adoption_benefit": "adoption benefit efficiency automation lower effort lower cost workload reduction simplification productivity",
    "adoption_burden": "adoption burden staffing skills expertise resources cost funding training maintenance procurement migration overhead",
    "regulatory_applicability": "legal regulatory compliance applicability requirement cybersecurity control",
    "governance_enablement": "governance auditability accountability traceability documentation assessment framework transparency management",
    "compliance_constraint": "compliance legal regulatory privacy requirement restriction mandatory obligation notification localization",
}


_SLOT_PROXIMITY_TERMS: dict[str, tuple[str, ...]] = {
    "technical_enablement": (
        "can be",
        "may be",
        "allow",
        "enable",
        "support",
        "facilitat",
        "improv",
        "automati",
        "reference implementation",
        "provides",
        "achieve",
    ),
    "technical_limitation": (
        "limitation",
        "constraint",
        "challenge",
        "drawback",
        "false positive",
        "false negative",
        "latency",
        "overhead",
        "complex",
        "difficult",
        "failure",
        "weakness",
        "risk",
        "incompatib",
    ),
    "deployment_maturity": (
        "in production",
        "deployed",
        "production deployment",
        "operational deployment",
        "real-world",
        "real world",
        "case study",
        "reference implementation",
        "implemented",
        "currently used",
        "adopted",
    ),
    "adoption_benefit": (
        "cost-effective",
        "reduce cost",
        "lower cost",
        "reduce effort",
        "lower effort",
        "efficien",
        "save",
        "simplif",
        "reduce workload",
        "productiv",
        "resource saving",
        "time saving",
    ),
    "adoption_burden": (
        "staff",
        "personnel",
        "skill",
        "expertise",
        "resource",
        "training",
        "cost",
        "funding",
        "maintain",
        "maintenance",
        "burden",
        "overhead",
        "procure",
        "investment",
        "migration",
        "complex",
    ),
    "regulatory_applicability": (
        "legal",
        "regulat",
        "compliance",
        "requirement",
        "shall",
        "must",
        "policy",
        "directive",
    ),
    "governance_enablement": (
        "governance",
        "audit",
        "assessment",
        "document",
        "accountab",
        "traceab",
        "transparen",
        "framework",
        "manage",
    ),
    "compliance_constraint": (
        "compliance",
        "legal",
        "regulat",
        "privacy",
        "shall",
        "must",
        "required",
        "requirement",
        "restriction",
        "mandatory",
        "obligation",
        "notification",
        "localization",
    ),
}

_SEMANTIC_PROXIMITY_WINDOW_CHARS = 420


_STRICT_CURATED_SLOT_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "adoption_benefit": tuple(
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"\breduc(?:e|es|ed|ing)\s+(?:costs?|effort|workload)\b",
            r"\breduc(?:e|es|ed|ing)\s+(?:manual\s+)?(?:review|monitoring|administrative)\s+(?:time|effort|workload)\b",
            r"\blower\s+(?:costs?|effort)\b",
            r"\b(?:improv(?:e|es|ed|ing)|increas(?:e|es|ed|ing))\s+(?:development\s+|operational\s+)?efficien(?:cy|cies)\b",
            r"\bincreas(?:e|es|ed|ing)\s+visibility\b",
            r"\breal[- ]time\s+alert(?:ing|s)\b",
            r"\b(?:time|resource)\s+sav(?:ing|ings)\b",
            r"\bsimplif(?:y|ies|ied|ication)\s+(?:the\s+)?(?:management|administration|operations?)\b",
            r"\breduc(?:e|es|ed|ing)\s+(?:size|overhead)\b",
            r"\bcost[- ]effective\b.{0,120}\b(?:implementation|deployment|operation|monitoring|management)\b",
            r"\b(?:implementation|deployment|operation|monitoring|management)\b.{0,120}\bcost[- ]effective\b",
        )
    ),
    "adoption_burden": tuple(
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"\bsignificant\s+resources\b",
            r"\bresources\s+(?:needed|required)\b",
            r"\bresource[- ]intensive\b",
            r"\b(?:requires?|needs?)\b.{0,100}\b(?:staffing|personnel|funding|skills?|expertise|resources)\b",
            r"\b(?:staffing|personnel|funding|skills?|expertise|resources)\b.{0,100}\b(?:required|needed)\b",
            r"\b(?:implementation|deployment|operational|operation|maintenance|migration|procurement|training|staffing)\s+costs?\b",
            r"\bcosts?\b.{0,120}\b(?:implement|deploy|operate|maintain|migrate|procure|train|staff)\w*\b",
            r"\bcostly\b.{0,120}\b(?:implement|deploy|operate|maintain|migrate|procure)\w*\b",
            r"\bmaintenance\s+(?:costs?|burden|requirements?)\b",
            r"\bprocurement\s+(?:costs?|burden|effort|requirements?)\b",
            r"\bmigration\s+(?:costs?|effort|burden)\b",
            r"\b(?:computational|memory|processing|operational|administrative)\s+overhead\b",
            r"\bmust\s+invest\b",
            r"\brequires?\s+(?:an?\s+)?investment\b",
            r"\btraining\s+(?:requirements?|costs?|burden)\b",
        )
    ),
    "governance_enablement": tuple(
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"\b(?:establish|implement|maintain|document|monitor|track|audit|assess|review|verify)\w*\b.{0,180}\b(?:governance|accountab|traceab|audit|monitor|polic(?:y|ies)|controls?|risk management|security measures?)\b",
            r"\b(?:governance|accountab|traceab|audit|monitor|polic(?:y|ies)|controls?|risk management|security measures?)\b.{0,180}\b(?:establish|implement|maintain|document|monitor|track|audit|assess|review|verify)\w*\b",
            r"\btransparen(?:cy|t)\b.{0,100}\baccountab(?:ility|le)\b",
            r"\baccountab(?:ility|le)\b.{0,100}\btransparen(?:cy|t)\b",
        )
    ),
    "compliance_constraint": tuple(
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"\bcompliance\s+with\b.{0,180}\b(?:legal|regulatory|privacy|requirements?|standards?|specifications?)\b",
            r"\b(?:legal|regulatory|privacy)\b.{0,180}\b(?:must|shall|required|requirement|restrictions?|mandatory|obligation|compliance)\b",
            r"\b(?:must|shall|required|mandatory|obligation)\b.{0,180}\b(?:legal|regulatory|privacy|requirements?|controls?|standards?)\b",
        )
    ),
    "deployment_maturity": tuple(
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"\bin\s+production\b",
            r"\b(?:has|have|had|was|were|is|are)\s+deployed\b",
            r"\bdeployed\s+in\b",
            r"\bproduction\s+deployment\b",
            r"\boperational\s+deployment\b",
            r"\bcase\s+stud(?:y|ies)\b",
            r"\breference\s+implementation\b",
            r"\bimplemented\s+in\b",
            r"\bcurrently\s+used\b",
            r"\badopted\s+by\b",
        )
    ),
}


# Curated guidance documents are bound to a PMT at source-freeze time, but not
# every page/chunk in a long guidance document is valid evidence for every
# decision facet.  This allowlist prevents generic front matter or unrelated
# sections from entering a directional slot merely because they contain a broad
# keyword.
_CURATED_GUIDANCE_SLOT_ALLOWLIST: dict[str, frozenset[str]] = {
    "nist-ai-rmf-1-0": frozenset({"adoption_burden", "governance_enablement", "compliance_constraint", "deployment_maturity"}),
    "nist-ai-600-1": frozenset({"adoption_burden", "governance_enablement", "compliance_constraint", "deployment_maturity"}),
    "nist-sp-800-218a": frozenset({"adoption_benefit", "adoption_burden", "deployment_maturity"}),
    "ncsc-guidelines-secure-ai-system-development": frozenset({"adoption_burden", "governance_enablement", "compliance_constraint", "deployment_maturity"}),
    "nist-sp-800-94": frozenset({"adoption_benefit", "adoption_burden", "governance_enablement", "deployment_maturity"}),
    "nist-ir-8219": frozenset({"adoption_benefit", "adoption_burden", "governance_enablement", "deployment_maturity"}),
    "nist-sp-1800-10": frozenset({"adoption_benefit", "adoption_burden", "governance_enablement", "deployment_maturity"}),
    "nist-sp-800-137": frozenset({"adoption_benefit", "adoption_burden", "governance_enablement", "deployment_maturity"}),
    "cisa-ics-defense-in-depth-2016": frozenset({"adoption_benefit", "adoption_burden", "governance_enablement", "deployment_maturity"}),
    "nist-sp-800-57pt1r5": frozenset({"adoption_burden", "governance_enablement", "compliance_constraint", "deployment_maturity"}),
    "nist-sp-800-131ar2": frozenset({"compliance_constraint", "deployment_maturity"}),
    "nist-sp-800-52r2": frozenset({"adoption_benefit", "adoption_burden", "governance_enablement", "compliance_constraint", "deployment_maturity"}),
    "cisa-dhs-control-systems-encryption-primer": frozenset({"adoption_benefit", "adoption_burden", "governance_enablement", "deployment_maturity"}),
    "nist-sp-800-162": frozenset({"adoption_benefit", "adoption_burden", "governance_enablement", "deployment_maturity"}),
    "nist-sp-800-207": frozenset({"adoption_benefit", "adoption_burden", "governance_enablement", "deployment_maturity"}),
    "nist-sp-800-63b": frozenset({"adoption_benefit", "adoption_burden", "governance_enablement", "compliance_constraint", "deployment_maturity"}),
    "cisa-zero-trust-maturity-model-v2": frozenset({"adoption_benefit", "adoption_burden", "governance_enablement", "deployment_maturity"}),
    "nist-sp-800-113": frozenset({"adoption_benefit", "adoption_burden", "governance_enablement", "deployment_maturity"}),
}


def _record_matches_strict_curated_slot(record: EvidenceRecord, *, slot: str) -> bool:
    allowed_slots = _CURATED_GUIDANCE_SLOT_ALLOWLIST.get(record.source_document_id)
    if allowed_slots is not None and slot not in allowed_slots:
        return False
    lowered = record.content.lower()
    # Reject publication boilerplate, reference lists, acronym appendices, and
    # table-of-contents/figure-list chunks before any directional keyword test.
    if "reports on computer systems technology" in lowered and "information technology laboratory" in lowered:
        return False
    if "appendix b" in lowered and "acronym" in lowered:
        return False
    if "selected acronyms" in lowered:
        return False
    if lowered.count("https://doi.org") >= 3:
        return False
    if ("figures figure 1" in lowered or "list of figures" in lowered) and lowered.count("...") >= 3:
        return False
    if slot == "adoption_benefit":
        # NIST front matter describes ITL's statutory mission using generic
        # phrases such as "cost-effective security" and "productive use of
        # information technology". Those are publication boilerplate, not an
        # adoption benefit of the candidate PMT.
        if "itl’s responsibilities include" in lowered or "itl's responsibilities include" in lowered:
            return False
    patterns = _STRICT_CURATED_SLOT_PATTERNS.get(slot)
    if not patterns:
        return True
    return any(pattern.search(record.content) for pattern in patterns)


def _candidate_records(
    store: EvidenceStore,
    *,
    slot: str,
    cutoff_date: str,
    jurisdiction_source_registry_id: str,
    threat_id: str,
    pmt_id: str,
    pmt_name: str,
    supplemental_records: tuple[EvidenceRecord, ...] = (),
    require_action_specific_technical: bool = False,
) -> list[EvidenceRecord]:
    cutoff = date.fromisoformat(cutoff_date)
    allowed_types = DECISION_EVIDENCE_SLOT_CLAIM_TYPES[slot]
    candidates: list[EvidenceRecord] = []
    for record in (*store.records, *supplemental_records):
        if date.fromisoformat(record.available_at) > cutoff:
            continue
        if not (set(record.allowed_claim_types) & allowed_types):
            continue
        if record.retrieval_role == "decision_action_curated":
            # Action findings are manually bound to one frozen Threat×PMT pair.
            # They may carry only the Technical support/challenge slots.
            if threat_id not in record.threat_ids or pmt_id not in record.pmt_ids:
                continue
            if slot not in {"technical_enablement", "technical_limitation"}:
                continue
        elif require_action_specific_technical and slot in {"technical_enablement", "technical_limitation"}:
            # v5 forbids generic PMT guidance from manufacturing Technical
            # direction for a specific Threat×PMT action on the main path.
            # Fallback/ablation retrieval remains available when the frozen
            # action registry is intentionally not supplied.
            continue
        if slot == "regulatory_applicability":
            if record.source_registry_id != jurisdiction_source_registry_id:
                continue
        elif record.source_registry_id in {"30", "31", "32"}:
            # Main jurisdiction sweep changes only the explicit regulatory slot.
            # Technical/resource retrieval is therefore identical across KR/EU/US.
            continue
        if not _record_passes_source_quality_gate(record):
            continue
        if not _record_is_pmt_relevant(record, pmt_id=pmt_id, pmt_name=pmt_name):
            continue
        if not _record_is_slot_relevant(
            record,
            slot=slot,
            threat_id=threat_id,
            pmt_id=pmt_id,
            pmt_name=pmt_name,
        ):
            continue
        candidates.append(record)
    return candidates


def _term_pattern(term: str, *, pmt_id: str | None = None) -> re.Pattern[str]:
    suffix = r"(?![A-Za-z0-9])"
    if pmt_id == "PMT_HTTPS" and term.strip().lower() == "https":
        # A URL scheme is provenance/navigation text, not evidence about HTTPS
        # as the candidate defensive technology.
        suffix = r"(?![A-Za-z0-9]|://)"
    return re.compile(
        r"(?<![A-Za-z0-9])" + re.escape(term.strip()) + suffix,
        re.IGNORECASE,
    )


def _pmt_terms(*, pmt_id: str, pmt_name: str) -> tuple[str, ...]:
    terms = {pmt_name.strip()}
    terms.update(_PMT_DECISION_ALIASES.get(pmt_id, ()))
    return tuple(sorted(term for term in terms if len(term.strip()) >= 3))


def _record_is_pmt_relevant(record: EvidenceRecord, *, pmt_id: str, pmt_name: str) -> bool:
    """Hard semantic gate before ranking.

    Stage-4 decision retrieval must never manufacture slot coverage by ranking
    a record that is only lexically similar to the slot wording. The source text
    itself must contain the canonical PMT name or a narrow, predeclared synonym.
    Existing Stage-0 PMT tags remain a ranking tiebreaker, not an eligibility
    shortcut, because long chunks and navigation text can create false tags.
    """

    if record.retrieval_role == "decision_action_curated":
        return pmt_id in record.pmt_ids
    if record.retrieval_role == "decision_guidance_curated":
        # Curated Stage-4 guidance is bound to an explicit PMT at source-freeze
        # time.  Do not let incidental vocabulary inside a document mapped to a
        # different PMT re-enter through the generic lexical fallback.
        return pmt_id in record.pmt_ids
    return any(
        _term_pattern(term, pmt_id=pmt_id).search(record.content)
        for term in _pmt_terms(pmt_id=pmt_id, pmt_name=pmt_name)
    )


def _record_passes_source_quality_gate(record: EvidenceRecord) -> bool:
    # Source 10 is an operational vulnerability/advisory corpus. Its product
    # names and repeated mitigation boilerplate are useful to Stage 0/1 but are
    # too ambiguous to stand in for generic deployment/adoption evidence.
    if record.source_registry_id == "10":
        return False
    # CISA CPG PDF chunks also contain glossary/appendix material. For Stage 4
    # action evidence, keep only chunks that actually carry a Recommended Action.
    if record.source_registry_id == "23" and "RECOMMENDED ACTION" not in record.content.upper():
        return False
    return True


def _record_is_slot_relevant(
    record: EvidenceRecord,
    *,
    slot: str,
    threat_id: str,
    pmt_id: str,
    pmt_name: str,
) -> bool:
    """Require PMT and slot semantics to co-occur locally, not just in one long chunk."""

    content = record.content.lower()
    slot_terms = _SLOT_PROXIMITY_TERMS[slot]
    if record.retrieval_role == "decision_action_curated":
        if threat_id not in record.threat_ids or pmt_id not in record.pmt_ids:
            return False
        if slot == "technical_enablement":
            return "action_effectiveness" in record.allowed_claim_types
        if slot == "technical_limitation":
            return "action_limitation" in record.allowed_claim_types
        return False
    if record.retrieval_role == "decision_guidance_curated" and pmt_id in record.pmt_ids:
        # Curated decision-guidance documents are manually bound to a PMT at the
        # document level.  The chunk still has to carry the requested slot
        # semantics; the curated mapping replaces only the fragile repeated-PMT-
        # token requirement inside every individual chunk.
        if not _record_matches_strict_curated_slot(record, slot=slot):
            return False
        return any(term in content for term in slot_terms)
    matches: list[tuple[int, int]] = []
    for term in _pmt_terms(pmt_id=pmt_id, pmt_name=pmt_name):
        for match in _term_pattern(term, pmt_id=pmt_id).finditer(record.content):
            matches.append((match.start(), match.end()))
    if not matches:
        return False
    for start, end in matches:
        left = max(0, start - _SEMANTIC_PROXIMITY_WINDOW_CHARS)
        right = min(len(content), end + _SEMANTIC_PROXIMITY_WINDOW_CHARS)
        window = content[left:right]
        if any(term in window for term in slot_terms):
            return True
    return False


def _publisher_group(record: EvidenceRecord) -> str:
    """Return a deterministic organization-level source group.

    This is used only for diversity floors and ranking. It does not upgrade the
    evidentiary meaning of a record. Government primary-law sources retain
    jurisdiction-specific groups instead of being collapsed into one bucket.
    """

    source = record.source.upper()
    registry_id = record.source_registry_id.upper()
    if record.retrieval_role == "decision_action_curated":
        return {
            "AE01S": "SPRINGER_EAI",
            "AE01C": "SPRINGER_EAI",
            "AE02S": "ACM",
            "AE02C": "ACM",
            "AE03S": "OXFORD_ACADEMIC",
            "AE03C": "MDPI",
            "AE04S": "ELSEVIER",
            "AE04C": "ELSEVIER",
            "AE05S": "UNTAN_JUSTIN",
            "AE05C": "IPSJ",
            "AE06S": "ELSEVIER",
            "AE06C": "ELSEVIER",
            "AE07S": "EDP_SCIENCES",
            "AE07C": "SCITEPRESS",
        }.get(registry_id, f"ACTION_{registry_id}")
    if "NCSC" in source or registry_id == "DE15":
        return "NCSC_UK"
    if "ENISA" in source:
        return "ENISA"
    if "CISA" in source or "ICS-CERT" in source or "DEPARTMENT OF HOMELAND SECURITY" in source or registry_id in {
        "DE16",
        "DE17",
        "DE18",
    }:
        return "CISA_DHS"
    if "NIST" in source or registry_id.startswith("DE"):
        return "NIST"
    if record.source_registry_id == "30":
        return "KR_PRIMARY"
    if record.source_registry_id == "31":
        return "EU_PRIMARY"
    if record.source_registry_id == "32":
        return "US_PRIMARY"
    return f"SOURCE_{record.source_registry_id}"


def build_decision_evidence_pack(
    *,
    store: EvidenceStore,
    cutoff_date: str,
    threat_id: str,
    pmt_id: str,
    threat_name: str,
    pmt_name: str,
    scenario: dict[str, Any],
    supplemental_records: tuple[EvidenceRecord, ...] = (),
    supplemental_manifest: dict[str, Any] | None = None,
    action_records: tuple[EvidenceRecord, ...] = (),
    action_manifest: dict[str, Any] | None = None,
    main_guidance_records: tuple[EvidenceRecord, ...] = (),
    main_guidance_slot_coverage: dict[str, tuple[str, ...]] | None = None,
    main_guidance_manifest: dict[str, Any] | None = None,
    minimum_generic_records: int = 0,
    minimum_generic_documents: int = 0,
    minimum_lens_documents: int = 0,
    minimum_balance_documents_per_facet: int = 0,
    minimum_balance_publishers_per_facet: int = 0,
    minimum_lens_publishers: int = 0,
    max_records_per_slot: int = MAX_RECORDS_PER_DECISION_SLOT,
) -> dict[str, Any]:
    """Retrieve one deterministic, balance-gated Stage-4 evidence supplement."""

    if not 1 <= int(max_records_per_slot) <= MAX_RECORDS_PER_DECISION_SLOT:
        raise Stage0ValidationError(
            f"max_records_per_slot must be between 1 and {MAX_RECORDS_PER_DECISION_SLOT}"
        )

    if supplemental_records:
        if not supplemental_manifest:
            raise Stage0ValidationError("Supplemental decision records require a frozen supplement manifest")
        missing_manifest_fields = [
            key
            for key in ("snapshot_id", "manifest_sha256", "records_sha256")
            if not supplemental_manifest.get(key)
        ]
        if missing_manifest_fields:
            raise Stage0ValidationError(
                f"Supplemental decision manifest is missing required fields: {missing_manifest_fields}"
            )
    if action_records:
        if not action_manifest:
            raise Stage0ValidationError("Action evidence records require a frozen action-evidence manifest")
        missing_action_fields = [
            key
            for key in ("registry_version", "snapshot_id", "registry_sha256")
            if not action_manifest.get(key)
        ]
        if missing_action_fields:
            raise Stage0ValidationError(
                f"Action-evidence manifest is missing required fields: {missing_action_fields}"
            )
    if main_guidance_records:
        if not main_guidance_manifest or main_guidance_slot_coverage is None:
            raise Stage0ValidationError(
                "Main-guidance records require a frozen selection manifest and exact slot coverage"
            )
        missing_guidance_fields = [
            key
            for key in ("selection_version", "selection_sha256", "pmt_id")
            if not main_guidance_manifest.get(key)
        ]
        if missing_guidance_fields:
            raise Stage0ValidationError(
                f"Main-guidance manifest is missing required fields: {missing_guidance_fields}"
            )
        if str(main_guidance_manifest.get("pmt_id")) != pmt_id:
            raise Stage0ValidationError("Main-guidance manifest PMT mismatch")
        guidance_ids = {record.evidence_id for record in main_guidance_records}
        if set(main_guidance_slot_coverage) != guidance_ids:
            raise Stage0ValidationError(
                "Main-guidance exact slot coverage must match selected records exactly"
            )
        valid_guidance_slots = {
            "deployment_maturity",
            "adoption_benefit",
            "adoption_burden",
            "governance_enablement",
            "compliance_constraint",
        }
        for evidence_id, slots in main_guidance_slot_coverage.items():
            if not slots or set(slots) - valid_guidance_slots:
                raise Stage0ValidationError(
                    f"Main-guidance selection {evidence_id} has invalid slots: {slots}"
                )

    jurisdiction_source_id = str(scenario["jurisdiction_source_registry_id"])
    selected_by_id: dict[str, EvidenceRecord] = {}
    coverage: dict[str, set[str]] = {}
    queries: dict[str, str] = {}
    scores_by_slot: dict[str, dict[str, float]] = {}
    candidate_counts: dict[str, int] = {}
    lens_seen_documents: dict[str, set[str]] = {
        lens_name: set() for lens_name in MAIN_LENS_DECISION_EVIDENCE_SLOTS
    }

    for slot in DECISION_EVIDENCE_SLOT_CLAIM_TYPES:
        if (
            main_guidance_records
            and slot
            not in {
                "technical_enablement",
                "technical_limitation",
                "regulatory_applicability",
            }
        ):
            chosen = [
                record
                for record in main_guidance_records
                if slot in set(main_guidance_slot_coverage.get(record.evidence_id, ()))
            ]
            candidate_counts[slot] = len(chosen)
            queries[slot] = "EXACT_FROZEN_MAIN_GUIDANCE_SELECTION"
            scores_by_slot[slot] = {}
            slot_lenses = {
                lens_name
                for lens_name, eligible_slots in MAIN_LENS_DECISION_EVIDENCE_SLOTS.items()
                if slot in eligible_slots
            }
            for record in chosen:
                selected_by_id[record.evidence_id] = record
                coverage.setdefault(record.evidence_id, set()).add(slot)
                for lens_name in slot_lenses:
                    lens_seen_documents[lens_name].add(record.source_document_id)
            continue
        candidates = _candidate_records(
            store,
            slot=slot,
            cutoff_date=cutoff_date,
            jurisdiction_source_registry_id=jurisdiction_source_id,
            threat_id=threat_id,
            pmt_id=pmt_id,
            pmt_name=pmt_name,
            supplemental_records=(*supplemental_records, *action_records),
            require_action_specific_technical=bool(action_records),
        )
        candidate_counts[slot] = len(candidates)
        jurisdiction_term = scenario["region"] if slot == "regulatory_applicability" else ""
        query = " ".join(
            part
            for part in (threat_name, pmt_name, jurisdiction_term, _SLOT_QUERY_HINTS[slot])
            if part
        )
        queries[slot] = query
        ranked_scores = rank_bm25([record.content for record in candidates], query)
        score_by_id = {candidates[item.index].evidence_id: item.score for item in ranked_scores}
        scores_by_slot[slot] = {
            evidence_id: round(score, 12)
            for evidence_id, score in sorted(score_by_id.items())
        }
        slot_lenses = {
            lens_name
            for lens_name, eligible_slots in MAIN_LENS_DECISION_EVIDENCE_SLOTS.items()
            if slot in eligible_slots
        }
        ranked = sorted(
            candidates,
            key=lambda record: (
                -sum(
                    int(record.source_document_id not in lens_seen_documents[lens_name])
                    for lens_name in slot_lenses
                ),
                -score_by_id.get(record.evidence_id, 0.0),
                -(int(pmt_id in record.pmt_ids) + int(threat_id in record.threat_ids)),
                record.source_family,
                record.source_registry_id,
                record.source_document_id,
                record.evidence_id,
            ),
        )
        chosen: list[EvidenceRecord] = []
        used_documents: set[str] = set()
        used_publishers: set[str] = set()
        # First pass explicitly prefers independent publishing organizations.
        for record in ranked:
            publisher = _publisher_group(record)
            if record.source_document_id in used_documents or publisher in used_publishers:
                continue
            chosen.append(record)
            used_documents.add(record.source_document_id)
            used_publishers.add(publisher)
            if len(chosen) == max_records_per_slot:
                break
        # Second pass preserves document diversity even when only one publisher
        # contains enough material for a specific facet.
        if len(chosen) < max_records_per_slot:
            for record in ranked:
                if record in chosen or record.source_document_id in used_documents:
                    continue
                chosen.append(record)
                used_documents.add(record.source_document_id)
                if len(chosen) == max_records_per_slot:
                    break
        # Last resort keeps deterministic top-k behavior for sparse ablations;
        # the main experiment's balance floors below still reject shallow input.
        if len(chosen) < max_records_per_slot:
            for record in ranked:
                if record in chosen:
                    continue
                chosen.append(record)
                if len(chosen) == max_records_per_slot:
                    break
        for record in chosen:
            selected_by_id[record.evidence_id] = record
            coverage.setdefault(record.evidence_id, set()).add(slot)
            for lens_name in slot_lenses:
                lens_seen_documents[lens_name].add(record.source_document_id)

    selected = sorted(selected_by_id.values(), key=lambda record: record.evidence_id)
    context_record = cis_ig2_context_evidence_record()
    covered_slots = sorted({slot for slots in coverage.values() for slot in slots})
    generic_ids = {
        evidence_id
        for evidence_id, slots in coverage.items()
        if any(slot != "regulatory_applicability" for slot in slots)
    }
    generic_records = [record for record in selected if record.evidence_id in generic_ids]
    generic_documents = {record.source_document_id for record in generic_records}
    selected_by_evidence_id = {record.evidence_id: record for record in selected}
    lens_coverage: dict[str, dict[str, Any]] = {}
    for lens_name, eligible_slots in MAIN_LENS_DECISION_EVIDENCE_SLOTS.items():
        lens_ids = {
            evidence_id
            for evidence_id, slots in coverage.items()
            if set(slots) & eligible_slots
        }
        lens_documents = {
            selected_by_evidence_id[evidence_id].source_document_id
            for evidence_id in lens_ids
        }
        lens_publishers = {
            _publisher_group(selected_by_evidence_id[evidence_id])
            for evidence_id in lens_ids
        }
        facet_coverage: dict[str, dict[str, Any]] = {}
        for facet in sorted(MAIN_REQUIRED_BALANCE_FACETS_BY_LENS[lens_name]):
            facet_ids = {
                evidence_id
                for evidence_id, slots in coverage.items()
                if facet in slots
            }
            facet_documents = {
                selected_by_evidence_id[evidence_id].source_document_id
                for evidence_id in facet_ids
            }
            facet_publishers = {
                _publisher_group(selected_by_evidence_id[evidence_id])
                for evidence_id in facet_ids
            }
            facet_coverage[facet] = {
                "record_count": len(facet_ids),
                "document_count": len(facet_documents),
                "document_ids": sorted(facet_documents),
                "publisher_count": len(facet_publishers),
                "publisher_groups": sorted(facet_publishers),
            }
        lens_coverage[lens_name] = {
            "record_count": len(lens_ids),
            "document_count": len(lens_documents),
            "document_ids": sorted(lens_documents),
            "publisher_count": len(lens_publishers),
            "publisher_groups": sorted(lens_publishers),
            "eligible_slots": sorted(eligible_slots),
            "required_balance_facets": sorted(MAIN_REQUIRED_BALANCE_FACETS_BY_LENS[lens_name]),
            "facet_coverage": facet_coverage,
        }
    action_direction_coverage: dict[str, dict[str, Any]] = {}
    for facet in ("technical_enablement", "technical_limitation"):
        facet_ids = {
            evidence_id
            for evidence_id, slots in coverage.items()
            if facet in slots
        }
        action_ids = {
            evidence_id
            for evidence_id in facet_ids
            if selected_by_evidence_id[evidence_id].retrieval_role == "decision_action_curated"
            and threat_id in selected_by_evidence_id[evidence_id].threat_ids
            and pmt_id in selected_by_evidence_id[evidence_id].pmt_ids
        }
        action_direction_coverage[facet] = {
            "record_count": len(action_ids),
            "evidence_ids": sorted(action_ids),
            "document_ids": sorted(
                {selected_by_evidence_id[evidence_id].source_document_id for evidence_id in action_ids}
            ),
        }
    coverage_issues: list[str] = []
    if len(generic_records) < minimum_generic_records:
        coverage_issues.append(
            f"generic decision evidence records={len(generic_records)} < required {minimum_generic_records}"
        )
    if len(generic_documents) < minimum_generic_documents:
        coverage_issues.append(
            f"generic decision evidence documents={len(generic_documents)} < required {minimum_generic_documents}"
        )
    if minimum_lens_documents:
        for lens_name, lens_floor in lens_coverage.items():
            if lens_floor["document_count"] < minimum_lens_documents:
                coverage_issues.append(
                    f"{lens_name} decision evidence documents={lens_floor['document_count']} "
                    f"< required {minimum_lens_documents}"
                )
    if minimum_lens_publishers:
        for lens_name, lens_floor in lens_coverage.items():
            if lens_floor["publisher_count"] < minimum_lens_publishers:
                coverage_issues.append(
                    f"{lens_name} decision evidence publishers={lens_floor['publisher_count']} "
                    f"< required {minimum_lens_publishers}"
                )
    if action_records:
        for facet, action_floor in action_direction_coverage.items():
            if action_floor["record_count"] < MAIN_MIN_ACTION_DIRECTION_RECORDS:
                coverage_issues.append(
                    f"technical action-specific {facet} records={action_floor['record_count']} "
                    f"< required {MAIN_MIN_ACTION_DIRECTION_RECORDS}"
                )
    if minimum_balance_documents_per_facet:
        for lens_name, lens_floor in lens_coverage.items():
            for facet, facet_floor in lens_floor["facet_coverage"].items():
                if facet_floor["document_count"] < minimum_balance_documents_per_facet:
                    coverage_issues.append(
                        f"{lens_name}.{facet} documents={facet_floor['document_count']} "
                        f"< required {minimum_balance_documents_per_facet}"
                    )
    if minimum_balance_publishers_per_facet:
        for lens_name, lens_floor in lens_coverage.items():
            for facet, facet_floor in lens_floor["facet_coverage"].items():
                if facet_floor["publisher_count"] < minimum_balance_publishers_per_facet:
                    coverage_issues.append(
                        f"{lens_name}.{facet} publishers={facet_floor['publisher_count']} "
                        f"< required {minimum_balance_publishers_per_facet}"
                    )
    if coverage_issues:
        raise Stage0ValidationError(
            f"Stage-4 decision-evidence coverage floor failed for {pmt_id} / {scenario['scenario_id']}: "
            + "; ".join(coverage_issues)
        )
    return {
        "retrieval_version": DECISION_EVIDENCE_RETRIEVAL_VERSION,
        "scenario_id": scenario["scenario_id"],
        "source_snapshot_id": store.snapshot_id,
        "decision_source_snapshot_id": (
            str(supplemental_manifest.get("snapshot_id"))
            if supplemental_manifest
            else None
        ),
        "decision_source_manifest_sha256": (
            str(supplemental_manifest.get("manifest_sha256"))
            if supplemental_manifest
            else None
        ),
        "decision_source_records_sha256": (
            str(supplemental_manifest.get("records_sha256"))
            if supplemental_manifest
            else None
        ),
        "action_evidence_snapshot_id": (
            str(action_manifest.get("snapshot_id"))
            if action_manifest
            else None
        ),
        "action_evidence_registry_version": (
            str(action_manifest.get("registry_version"))
            if action_manifest
            else None
        ),
        "action_evidence_registry_sha256": (
            str(action_manifest.get("registry_sha256"))
            if action_manifest
            else None
        ),
        "main_guidance_selection_version": (
            str(main_guidance_manifest.get("selection_version"))
            if main_guidance_manifest
            else None
        ),
        "main_guidance_selection_sha256": (
            str(main_guidance_manifest.get("selection_sha256"))
            if main_guidance_manifest
            else None
        ),
        "analysis_cutoff_date": cutoff_date,
        "context_evidence_records": [context_record],
        "decision_evidence_records": [record.to_dict() for record in selected],
        "context_evidence_ids": [context_record["evidence_id"]],
        "decision_evidence_ids": [record.evidence_id for record in selected],
        "slot_coverage": {
            evidence_id: sorted(slots)
            for evidence_id, slots in sorted(coverage.items())
        },
        "retrieval_metadata": {
            "method": (
                "main_exact_guidance_allowlist_plus_action_specific_technical_plus_jurisdiction_regulatory"
                if main_guidance_records
                else "cutoff_filter_claim_contract_source_quality_gate_action_specific_technical_gate_strict_slot_semantics_bm25_document_diversity_top_k"
            ),
            "retriever_version": BM25_RETRIEVER_VERSION,
            "semantic_gate": {
                "rule": "predeclared_pmt_term_in_record_content_plus_slot_semantics_within_local_window",
                "pmt_id": pmt_id,
                "pmt_terms": list(_pmt_terms(pmt_id=pmt_id, pmt_name=pmt_name)),
                "slot_proximity_window_chars": _SEMANTIC_PROXIMITY_WINDOW_CHARS,
            },
            "source_quality_gate": {
                "excluded_source_registry_ids": ["10"],
                "cisa_cpg_source_registry_id": "23",
                "cisa_cpg_requirement": "record content contains RECOMMENDED ACTION",
            },
            "decision_source_supplement": {
                "snapshot_id": (
                    str(supplemental_manifest.get("snapshot_id"))
                    if supplemental_manifest
                    else None
                ),
                "manifest_sha256": (
                    str(supplemental_manifest.get("manifest_sha256"))
                    if supplemental_manifest
                    else None
                ),
                "records_sha256": (
                    str(supplemental_manifest.get("records_sha256"))
                    if supplemental_manifest
                    else None
                ),
                "record_count": len(supplemental_records),
            },
            "action_evidence_supplement": {
                "snapshot_id": (
                    str(action_manifest.get("snapshot_id"))
                    if action_manifest
                    else None
                ),
                "registry_version": (
                    str(action_manifest.get("registry_version"))
                    if action_manifest
                    else None
                ),
                "registry_sha256": (
                    str(action_manifest.get("registry_sha256"))
                    if action_manifest
                    else None
                ),
                "record_count": len(action_records),
                "direction_coverage": action_direction_coverage,
                "rule": "Technical direction requires exact frozen Threat×PMT action evidence; generic PMT guidance cannot carry Technical +/- direction.",
            },
            "main_guidance_selection": {
                "selection_version": (
                    str(main_guidance_manifest.get("selection_version"))
                    if main_guidance_manifest
                    else None
                ),
                "selection_sha256": (
                    str(main_guidance_manifest.get("selection_sha256"))
                    if main_guidance_manifest
                    else None
                ),
                "selected_record_count": len(main_guidance_records),
                "selection_mode": (
                    "exact_frozen_evidence_id_allowlist"
                    if main_guidance_records
                    else "retriever_fallback"
                ),
                "rule": "Main-run deployment/adoption/governance guidance is selected only by audited frozen evidence_id; BM25 remains fallback/ablation only.",
            },
            "coverage_floor": {
                "minimum_generic_records": minimum_generic_records,
                "minimum_generic_documents": minimum_generic_documents,
                "minimum_lens_documents": minimum_lens_documents,
                "minimum_balance_documents_per_facet": minimum_balance_documents_per_facet,
                "minimum_balance_publishers_per_facet": minimum_balance_publishers_per_facet,
                "minimum_lens_publishers": minimum_lens_publishers,
                "minimum_action_direction_records": MAIN_MIN_ACTION_DIRECTION_RECORDS,
                "generic_record_count": len(generic_records),
                "generic_document_count": len(generic_documents),
                "generic_document_ids": sorted(generic_documents),
                "lens_coverage": lens_coverage,
            },
            "bm25_parameters": {"k1": BM25_K1, "b": BM25_B},
            "max_records_per_slot": int(max_records_per_slot),
            "candidate_counts": candidate_counts,
            "queries": queries,
            "scores_by_slot": scores_by_slot,
            "required_slots": list(DECISION_EVIDENCE_SLOT_CLAIM_TYPES),
            "covered_slots": covered_slots,
            "missing_slots": sorted(set(DECISION_EVIDENCE_SLOT_CLAIM_TYPES) - set(covered_slots)),
            "jurisdiction_source_registry_id": jurisdiction_source_id,
            "generic_slots_are_jurisdiction_invariant": True,
        },
    }


def open_snapshot_for_decision_evidence(*, project_root: str | Path, snapshot_id: str) -> EvidenceStore:
    snapshot_dir = (
        Path(project_root).resolve()
        / "Multi-Agent"
        / "Results"
        / "Stage0"
        / "Evidence"
        / "snapshots"
        / snapshot_id
    )
    if not snapshot_dir.is_dir():
        raise Stage0ValidationError(f"Decision evidence snapshot is missing: {snapshot_dir}")
    return EvidenceStore(snapshot_dir)

