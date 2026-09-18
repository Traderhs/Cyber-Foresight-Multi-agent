from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
import hashlib
import json
from typing import Any, Iterable

from .schema import EvidenceRecord, Stage0ValidationError


MAIN_GUIDANCE_SELECTION_VERSION = "stage3-main-guidance-selection-v1"
MAIN_GUIDANCE_SELECTION_CUTOFF = "2024-12-31"
MAIN_GUIDANCE_SELECTION_SHA256 = "3feabfbd76fc1155bb85a720a0085166820478651d7ef5be8079a47bb2f1a072"


@dataclass(frozen=True)
class MainGuidanceSelectionSpec:
    pmt_id: str
    evidence_id: str
    source_document_id: str
    slots: tuple[str, ...]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["slots"] = list(self.slots)
        return value


# Main-run guidance is selected by exact frozen evidence_id rather than by
# lexical/BM25 ranking.  These records were audited before the v5 Stage-4 rerun
# for the narrow semantics assigned below.  Unequal support/challenge counts are
# intentional: the selection contract prevents retrieval bias; it does not
# manufacture a 50:50 evidence distribution.
MAIN_GUIDANCE_SELECTION_SPECS: tuple[MainGuidanceSelectionSpec, ...] = (
    # NLP / LLM
    MainGuidanceSelectionSpec(
        "PMT_NLP_LLM",
        "DE_DE01_E85CB6412740EF20",
        "nist-ai-rmf-1-0",
        ("adoption_burden",),
        "AI RMF explicitly identifies staffing/funding resources needed for AI risk-management goals.",
    ),
    MainGuidanceSelectionSpec(
        "PMT_NLP_LLM",
        "DE_DE03_6786A7AB93821D48",
        "nist-sp-800-218a",
        ("adoption_benefit",),
        "Automated vulnerability-analysis methods are stated to lower detection effort and resource needs.",
    ),
    MainGuidanceSelectionSpec(
        "PMT_NLP_LLM",
        "DE_DE03_912FC8A63179A943",
        "nist-sp-800-218a",
        ("adoption_benefit",),
        "Secure-by-design practice is explicitly described as improving development efficiency.",
    ),
    MainGuidanceSelectionSpec(
        "PMT_NLP_LLM",
        "DE_DE15_1AEC6DDD669756FF",
        "ncsc-guidelines-secure-ai-system-development",
        ("adoption_burden",),
        "NCSC states secure-by-design AI requires significant lifecycle resources and investment.",
    ),
    MainGuidanceSelectionSpec(
        "PMT_NLP_LLM",
        "DE_DE01_0B9728FE19095FD0",
        "nist-ai-rmf-1-0",
        ("governance_enablement",),
        "AI RMF GOVERN defines ongoing review plus explicit organizational roles/responsibilities.",
    ),
    MainGuidanceSelectionSpec(
        "PMT_NLP_LLM",
        "DE_DE02_007F05150CB46131",
        "nist-ai-600-1",
        ("compliance_constraint",),
        "GAI Profile explicitly requires incident reporting in accordance with applicable legal/regulatory requirements.",
    ),
    MainGuidanceSelectionSpec(
        "PMT_NLP_LLM",
        "DE_DE15_271AFADC7B3D5999",
        "ncsc-guidelines-secure-ai-system-development",
        ("governance_enablement",),
        "NCSC documentation guidance directly supports lifecycle transparency and accountability.",
    ),
    MainGuidanceSelectionSpec(
        "PMT_NLP_LLM",
        "DE_DE02_CFF4EAB2C4BEC522",
        "nist-ai-600-1",
        ("deployment_maturity",),
        "GAI Profile discusses monitoring deployed-system capabilities and limitations through TEVV.",
    ),
    MainGuidanceSelectionSpec(
        "PMT_NLP_LLM",
        "DE_DE15_7C1849B1FD93975C",
        "ncsc-guidelines-secure-ai-system-development",
        ("deployment_maturity", "governance_enablement"),
        "NCSC gives concrete post-deployment operation, logging, monitoring, update, and incident-management guidance.",
    ),

    # Anomaly Detection
    MainGuidanceSelectionSpec(
        "PMT_ANOMALY_DETECTION",
        "DE_DE05_105D0030778F41BC",
        "nist-ir-8219",
        ("adoption_benefit", "deployment_maturity", "governance_enablement"),
        "NIST BAD reference practice reports increased network visibility/real-time alerting and commercially supplied implementations.",
    ),
    MainGuidanceSelectionSpec(
        "PMT_ANOMALY_DETECTION",
        "DE_DE16_E455D97291B61E2A",
        "cisa-ics-defense-in-depth-2016",
        ("adoption_benefit",),
        "CISA describes automated SIEM/IDS integration reducing manual review effort and speeding analysis.",
    ),
    MainGuidanceSelectionSpec(
        "PMT_ANOMALY_DETECTION",
        "DE_DE04_C6F46FCDE66BFAC7",
        "nist-sp-800-94",
        ("adoption_burden",),
        "NIST requires testing IDPS software/signature updates and maintaining dedicated test capability.",
    ),
    MainGuidanceSelectionSpec(
        "PMT_ANOMALY_DETECTION",
        "DE_DE04_0B77B88875745973",
        "nist-sp-800-94",
        ("governance_enablement",),
        "NIST specifies ongoing IDPS administrative/security maintenance and protected management access.",
    ),
    MainGuidanceSelectionSpec(
        "PMT_ANOMALY_DETECTION",
        "DE_DE06_14CCCEB6F82351D5",
        "nist-sp-1800-10",
        ("governance_enablement",),
        "Reference implementation capability includes external audit-log review/analysis/reporting.",
    ),
    MainGuidanceSelectionSpec(
        "PMT_ANOMALY_DETECTION",
        "DE_DE16_CEA1C38E613068F8",
        "cisa-ics-defense-in-depth-2016",
        ("governance_enablement",),
        "CISA characterizes IDS as a warning/audit mechanism feeding broader event analysis.",
    ),
    MainGuidanceSelectionSpec(
        "PMT_ANOMALY_DETECTION",
        "DE_DE06_0380E777894ACF9A",
        "nist-sp-1800-10",
        ("deployment_maturity",),
        "NIST reference implementation records configured behavior-anomaly detection and functional test results.",
    ),

    # Cryptography (used by Ransomware and Insider Threat main cases)
    MainGuidanceSelectionSpec(
        "PMT_CRYPTOGRAPHY",
        "DE_DE10_A086B3193CD6440B",
        "nist-sp-800-52r2",
        ("adoption_benefit",),
        "TLS Raw Public Keys can reduce public-key structure size and simplify processing, with an explicit assurance tradeoff.",
    ),
    MainGuidanceSelectionSpec(
        "PMT_CRYPTOGRAPHY",
        "DE_DE18_CDC975345637E009",
        "cisa-dhs-control-systems-encryption-primer",
        ("adoption_benefit", "deployment_maturity"),
        "DHS/CISA explains deployed IPSec transport/tunnel modes and the lower overhead of transport mode.",
    ),
    MainGuidanceSelectionSpec(
        "PMT_CRYPTOGRAPHY",
        "E_23_534489D29C38B327",
        "cisa-cpg",
        ("adoption_burden", "governance_enablement", "deployment_maturity"),
        "CISA CPG calls for strong/agile encryption, updating weak algorithms, PQ planning, and notes OT latency/availability feasibility constraints.",
    ),
    MainGuidanceSelectionSpec(
        "PMT_CRYPTOGRAPHY",
        "DE_DE08_031120C53EF050F3",
        "nist-sp-800-57pt1r5",
        ("governance_enablement",),
        "Key-management accountability/traceability requires defined authorized entities and auditable key actions.",
    ),
    MainGuidanceSelectionSpec(
        "PMT_CRYPTOGRAPHY",
        "DE_DE09_6719137897848A9E",
        "nist-sp-800-131ar2",
        ("compliance_constraint",),
        "NIST explicitly classifies cryptographic algorithms as acceptable, deprecated, legacy-use, or disallowed.",
    ),
    MainGuidanceSelectionSpec(
        "PMT_CRYPTOGRAPHY",
        "DE_DE18_EAA59A9B7CEAD276",
        "cisa-dhs-control-systems-encryption-primer",
        ("deployment_maturity",),
        "DHS/CISA documents TLS/SSL as standardized modern transport-layer cryptographic implementation practice.",
    ),

    # IDS / IPS
    MainGuidanceSelectionSpec(
        "PMT_IDS_IPS",
        "DE_DE04_DB6B29C9BE1B0F05",
        "nist-sp-800-94",
        ("adoption_benefit",),
        "NIST states integrated IDPS consoles can provide significant administrator/user time savings and streamline work.",
    ),
    MainGuidanceSelectionSpec(
        "PMT_IDS_IPS",
        "DE_DE07_86DF83870E0A768D",
        "nist-sp-800-137",
        ("adoption_benefit",),
        "NIST states appropriately sampled monitoring can be an efficient and effective monitoring strategy.",
    ),
    MainGuidanceSelectionSpec(
        "PMT_IDS_IPS",
        "DE_DE16_E455D97291B61E2A",
        "cisa-ics-defense-in-depth-2016",
        ("adoption_benefit",),
        "CISA describes SIEM/IDS integration reducing manual-review effort and speeding/automating analysis.",
    ),
    MainGuidanceSelectionSpec(
        "PMT_IDS_IPS",
        "DE_DE04_C6F46FCDE66BFAC7",
        "nist-sp-800-94",
        ("adoption_burden",),
        "NIST requires operational testing of IDPS updates and dedicated update-testing capability.",
    ),
    MainGuidanceSelectionSpec(
        "PMT_IDS_IPS",
        "DE_DE04_0B77B88875745973",
        "nist-sp-800-94",
        ("governance_enablement",),
        "NIST defines ongoing IDPS administrative/security maintenance responsibilities.",
    ),
    MainGuidanceSelectionSpec(
        "PMT_IDS_IPS",
        "DE_DE07_E2C403C2FF05BA09",
        "nist-sp-800-137",
        ("governance_enablement",),
        "Organization-wide continuous monitoring is explicitly tied to risk-related decision making across governance tiers.",
    ),
    MainGuidanceSelectionSpec(
        "PMT_IDS_IPS",
        "DE_DE16_CEA1C38E613068F8",
        "cisa-ics-defense-in-depth-2016",
        ("governance_enablement",),
        "CISA describes IDS as a warning/audit mechanism whose events require broader analysis.",
    ),
    MainGuidanceSelectionSpec(
        "PMT_IDS_IPS",
        "DE_DE04_0B017D64401987E4",
        "nist-sp-800-94",
        ("deployment_maturity",),
        "NIST documents concrete multi-product IDPS integration patterns including shared consoles/data exchange.",
    ),

    # Access Control
    MainGuidanceSelectionSpec(
        "PMT_ACCESS_CONTROL",
        "DE_DE11_1ECA9795EEF89DFA",
        "nist-sp-800-162",
        ("adoption_burden",),
        "NIST states ABAC development/deployment/maintenance can exceed benefits and identifies retrofit/infrastructure/policy-management costs.",
    ),
    MainGuidanceSelectionSpec(
        "PMT_ACCESS_CONTROL",
        "DE_DE17_429D17A4E11A72C3",
        "cisa-zero-trust-maturity-model-v2",
        ("adoption_benefit", "adoption_burden", "deployment_maturity"),
        "CISA states zero trust can take years and add initial cost while enabling longer-term more prudent allocation of security investment.",
    ),
    MainGuidanceSelectionSpec(
        "PMT_ACCESS_CONTROL",
        "DE_DE12_CBF6FBB1313D37AF",
        "nist-sp-800-207",
        ("adoption_burden",),
        "NIST documents potentially extreme provider-switching costs and long policy-migration programs in ZTA.",
    ),
    MainGuidanceSelectionSpec(
        "PMT_ACCESS_CONTROL",
        "DE_DE11_1E4AAE06CC3795D7",
        "nist-sp-800-162",
        ("governance_enablement",),
        "ABAC guidance requires local policies to reflect enterprise policy and defined attribute agreements.",
    ),
    MainGuidanceSelectionSpec(
        "PMT_ACCESS_CONTROL",
        "DE_DE12_E6F93E4AB8722379",
        "nist-sp-800-207",
        ("governance_enablement",),
        "NIST defines policy-administrator/enforcement responsibilities for enabling, monitoring, and terminating access.",
    ),
    MainGuidanceSelectionSpec(
        "PMT_ACCESS_CONTROL",
        "DE_DE17_567DDA7CFBC62829",
        "cisa-zero-trust-maturity-model-v2",
        ("governance_enablement",),
        "CISA links visibility/analytics to policy decisions and automation/orchestration to governed response workflows.",
    ),
    MainGuidanceSelectionSpec(
        "PMT_ACCESS_CONTROL",
        "DE_DE13_055FC5E35FA61DD5",
        "nist-sp-800-63b",
        ("compliance_constraint",),
        "NIST authentication guidance normatively requires protected channels and constrains session handling.",
    ),
    MainGuidanceSelectionSpec(
        "PMT_ACCESS_CONTROL",
        "DE_DE12_6509E7A6D8467E38",
        "nist-sp-800-207",
        ("deployment_maturity",),
        "NIST provides a candidate-solution selection process tied to workflow/ecosystem fit for ZTA deployment.",
    ),

    # HTTPS / TLS
    MainGuidanceSelectionSpec(
        "PMT_HTTPS",
        "DE_DE10_A086B3193CD6440B",
        "nist-sp-800-52r2",
        ("adoption_benefit",),
        "TLS Raw Public Keys can reduce key-structure size and simplify processing, with explicit assurance tradeoffs.",
    ),
    MainGuidanceSelectionSpec(
        "PMT_HTTPS",
        "DE_DE18_CDC975345637E009",
        "cisa-dhs-control-systems-encryption-primer",
        ("adoption_benefit",),
        "DHS/CISA documents lower-overhead transport-mode cryptographic deployment relative to tunnel mode.",
    ),
    MainGuidanceSelectionSpec(
        "PMT_HTTPS",
        "DE_DE14_EDCA207577B4607A",
        "nist-sp-800-113",
        ("adoption_burden",),
        "NIST documents SSL VPN performance/fragmentation overhead and potential network slowdown.",
    ),
    MainGuidanceSelectionSpec(
        "PMT_HTTPS",
        "DE_DE10_4BE87A62A6676C99",
        "nist-sp-800-52r2",
        ("compliance_constraint",),
        "NIST TLS guidance normatively requires server support for security extensions mitigating known attacks.",
    ),
    MainGuidanceSelectionSpec(
        "PMT_HTTPS",
        "DE_DE13_055FC5E35FA61DD5",
        "nist-sp-800-63b",
        ("compliance_constraint",),
        "NIST authentication guidance requires sessions to use authenticated protected channels and forbids insecure fallback.",
    ),
    MainGuidanceSelectionSpec(
        "PMT_HTTPS",
        "DE_DE14_575D613F0598EF5E",
        "nist-sp-800-113",
        ("governance_enablement",),
        "NIST recommends operational logging/review plus prototype/pilot validation for SSL VPN deployments.",
    ),
    MainGuidanceSelectionSpec(
        "PMT_HTTPS",
        "DE_DE18_20A979A3238D1DE6",
        "cisa-dhs-control-systems-encryption-primer",
        ("governance_enablement",),
        "DHS/CISA describes SSL/TLS retrofit governance prerequisites including mutual authentication and trusted-CA deployment.",
    ),
    MainGuidanceSelectionSpec(
        "PMT_HTTPS",
        "DE_DE14_5FC183B8D7DD961A",
        "nist-sp-800-113",
        ("deployment_maturity",),
        "NIST provides a concrete real-world SSL VPN planning/design/implementation case study and prototype.",
    ),
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def main_guidance_selection_sha256() -> str:
    return hashlib.sha256(
        _canonical_json([spec.to_dict() for spec in MAIN_GUIDANCE_SELECTION_SPECS])
    ).hexdigest()


def validate_main_guidance_selection_spec() -> dict[str, Any]:
    actual_hash = main_guidance_selection_sha256()
    issues: list[str] = []
    allowed_slots = {
        "deployment_maturity",
        "adoption_benefit",
        "adoption_burden",
        "governance_enablement",
        "compliance_constraint",
    }
    required_pmts = {
        "PMT_NLP_LLM",
        "PMT_ANOMALY_DETECTION",
        "PMT_CRYPTOGRAPHY",
        "PMT_IDS_IPS",
        "PMT_ACCESS_CONTROL",
        "PMT_HTTPS",
    }
    seen_pmts = {spec.pmt_id for spec in MAIN_GUIDANCE_SELECTION_SPECS}
    if seen_pmts != required_pmts:
        issues.append(f"main guidance PMT coverage mismatch: {sorted(seen_pmts)}")
    identities = [(spec.pmt_id, spec.evidence_id) for spec in MAIN_GUIDANCE_SELECTION_SPECS]
    if len(identities) != len(set(identities)):
        issues.append("main guidance (pmt_id, evidence_id) selections must be unique")
    for spec in MAIN_GUIDANCE_SELECTION_SPECS:
        if not spec.slots or set(spec.slots) - allowed_slots:
            issues.append(f"{spec.evidence_id} has invalid slots {spec.slots}")
        if not spec.rationale.strip():
            issues.append(f"{spec.evidence_id} is missing rationale")
    if MAIN_GUIDANCE_SELECTION_SHA256 != "TO_BE_PINNED" and actual_hash != MAIN_GUIDANCE_SELECTION_SHA256:
        issues.append(
            "main guidance selection changed without a new version/hash: "
            f"{actual_hash} != {MAIN_GUIDANCE_SELECTION_SHA256}"
        )
    if issues:
        raise Stage0ValidationError("Main-guidance selection validation failed: " + "; ".join(issues))
    return {
        "selection_version": MAIN_GUIDANCE_SELECTION_VERSION,
        "cutoff": MAIN_GUIDANCE_SELECTION_CUTOFF,
        "selection_sha256": actual_hash,
        "pmt_count": len(seen_pmts),
        "record_count": len(MAIN_GUIDANCE_SELECTION_SPECS),
    }


def load_main_guidance_selection(
    *,
    pmt_id: str,
    records: Iterable[EvidenceRecord],
) -> tuple[list[EvidenceRecord], dict[str, tuple[str, ...]], dict[str, Any]]:
    manifest = validate_main_guidance_selection_spec()
    cutoff = date.fromisoformat(MAIN_GUIDANCE_SELECTION_CUTOFF)
    by_id = {record.evidence_id: record for record in records}
    selected_specs = [spec for spec in MAIN_GUIDANCE_SELECTION_SPECS if spec.pmt_id == pmt_id]
    if not selected_specs:
        raise Stage0ValidationError(f"No frozen main-guidance selection for {pmt_id}")
    selected: list[EvidenceRecord] = []
    slot_coverage: dict[str, tuple[str, ...]] = {}
    issues: list[str] = []
    for spec in selected_specs:
        record = by_id.get(spec.evidence_id)
        if record is None:
            issues.append(f"missing exact evidence_id {spec.evidence_id}")
            continue
        if record.source_document_id != spec.source_document_id:
            issues.append(
                f"{spec.evidence_id} source_document_id changed: "
                f"{record.source_document_id} != {spec.source_document_id}"
            )
            continue
        if date.fromisoformat(record.available_at) > cutoff:
            issues.append(f"{spec.evidence_id} is post-cutoff ({record.available_at})")
            continue
        if pmt_id not in record.pmt_ids:
            issues.append(f"{spec.evidence_id} is not bound to {pmt_id}")
            continue
        selected.append(record)
        slot_coverage[record.evidence_id] = tuple(sorted(spec.slots))
    if issues:
        raise Stage0ValidationError(
            f"Frozen main-guidance selection failed for {pmt_id}: " + "; ".join(issues)
        )
    manifest = {
        **manifest,
        "pmt_id": pmt_id,
        "selected_record_count": len(selected),
        "selected_evidence_ids": sorted(slot_coverage),
    }
    return sorted(selected, key=lambda item: item.evidence_id), slot_coverage, manifest


__all__ = [
    "MAIN_GUIDANCE_SELECTION_CUTOFF",
    "MAIN_GUIDANCE_SELECTION_SHA256",
    "MAIN_GUIDANCE_SELECTION_SPECS",
    "MAIN_GUIDANCE_SELECTION_VERSION",
    "load_main_guidance_selection",
    "main_guidance_selection_sha256",
    "validate_main_guidance_selection_spec",
]
