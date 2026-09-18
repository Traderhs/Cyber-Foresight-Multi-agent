from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
import hashlib
import json
from typing import Any

from .schema import EvidenceRecord, Stage0ValidationError


ACTION_EVIDENCE_REGISTRY_VERSION = "stage3-action-evidence-registry-v1"
ACTION_EVIDENCE_SNAPSHOT_ID = "stage3-action-evidence-2024-12-31-v1"
ACTION_EVIDENCE_CUTOFF = "2024-12-31"
ACTION_EVIDENCE_REGISTRY_SHA256 = "02a3d2a913953c45a8891dbeb88004404ddf4eda341b9c334f7b054afff36497"


@dataclass(frozen=True)
class ActionEvidenceSpec:
    record_id: str
    source_name: str
    source_document_id: str
    source_uri: str
    publication_date: str
    available_at: str
    publisher_group: str
    threat_id: str
    pmt_id: str
    direction: str
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# These are compact, source-grounded paraphrases rather than copied article text.
# They are frozen before the contextual Stage-4 rerun and are used only to answer
# the narrow question "does this Threat × PMT action itself have direct empirical
# support or a documented direct limitation?" Generic deployment/governance
# guidance remains in the separate decision-guidance snapshot.
ACTION_EVIDENCE_SPECS: tuple[ActionEvidenceSpec, ...] = (
    ActionEvidenceSpec(
        "AE01S",
        "Guastalla et al., Application of Large Language Models to DDoS Attack Detection",
        "guastalla-llm-ddos-2024",
        "https://eudl.eu/doi/10.1007/978-3-031-51630-6_6",
        "2024-02-05",
        "2024-02-05",
        "SPRINGER_EAI",
        "THREAT_DDOS",
        "PMT_NLP_LLM",
        "SUPPORT",
        (
            "Direct DDoS experiments compared large language models with a conventional neural-network baseline. "
            "Few-shot and fine-tuned LLM variants detected DDoS traffic on CICIDS 2017 and an Urban IoT dataset, "
            "with fine-tuned variants reporting roughly mid-90-percent accuracy and outperforming the compared MLP."
        ),
    ),
    ActionEvidenceSpec(
        "AE01C",
        "Guastalla et al., Application of Large Language Models to DDoS Attack Detection",
        "guastalla-llm-ddos-2024",
        "https://eudl.eu/doi/10.1007/978-3-031-51630-6_6",
        "2024-02-05",
        "2024-02-05",
        "SPRINGER_EAI",
        "THREAT_DDOS",
        "PMT_NLP_LLM",
        "CHALLENGE",
        (
            "The DDoS study does not establish turnkey production deployment: performance depended on prompt context or fine-tuning, "
            "and few-shot results were materially weaker on the harder Urban IoT setting. Its evidence is benchmark-based rather than "
            "an enterprise operational deployment or cost study."
        ),
    ),
    ActionEvidenceSpec(
        "AE02S",
        "Patel et al., AProctor - A practical on-device antidote for Android malware",
        "patel-aproctor-malware-2023",
        "https://doi.org/10.1145/3579375.3579386",
        "2023-03-13",
        "2023-03-13",
        "ACM",
        "THREAT_MALWARE",
        "PMT_ANOMALY_DETECTION",
        "SUPPORT",
        (
            "AProctor directly evaluates anomaly-oriented malware detection on Android and reports high malware detection rates, "
            "including detection of temporally different and previously unseen APKs. The work demonstrates that anomaly-style "
            "behavioral detection can be implemented as an on-device malware defense."
        ),
    ),
    ActionEvidenceSpec(
        "AE02C",
        "Patel et al., AProctor - A practical on-device antidote for Android malware",
        "patel-aproctor-malware-2023",
        "https://doi.org/10.1145/3579375.3579386",
        "2023-03-13",
        "2023-03-13",
        "ACM",
        "THREAT_MALWARE",
        "PMT_ANOMALY_DETECTION",
        "CHALLENGE",
        (
            "The same malware study identifies a direct deployment limitation: anomaly-based algorithms can consume substantial "
            "energy and computing resources on constrained devices. The proposed design therefore required feature reduction and "
            "server-side training before on-device use."
        ),
    ),
    ActionEvidenceSpec(
        "AE03S",
        "Hassanin and Martinovic, CipherTrace: automatic detection of ciphers from execution traces to neutralize ransomware",
        "ciphertrace-ransomware-2024",
        "https://academic.oup.com/cybersecurity/article/10/1/tyae008/7688556",
        "2024-06-06",
        "2024-06-06",
        "OXFORD_ACADEMIC",
        "THREAT_RANSOMWARE",
        "PMT_CRYPTOGRAPHY",
        "SUPPORT",
        (
            "CipherTrace directly uses cryptographic-primitive analysis against ransomware. It identifies cipher classes and extracts "
            "cryptographic keys or state from execution traces across evaluated ransomware specimens, showing that cryptographic "
            "analysis can materially support ransomware neutralization and reverse engineering."
        ),
    ),
    ActionEvidenceSpec(
        "AE03C",
        "Al-rimy et al., Entropy Sharing in Ransomware: Bypassing Entropy-Based Detection of Cryptographic Operations",
        "entropy-sharing-ransomware-2024",
        "https://www.mdpi.com/1424-8220/24/5/1446",
        "2024-02-23",
        "2024-02-23",
        "MDPI",
        "THREAT_RANSOMWARE",
        "PMT_CRYPTOGRAPHY",
        "CHALLENGE",
        (
            "Ransomware can deliberately alter the observable entropy pattern of its cryptographic operations to evade detectors that "
            "treat high entropy as the key signal. The study demonstrates that cryptography-related detection is not universally robust "
            "and that some existing approaches also impose nontrivial computational cost."
        ),
    ),
    ActionEvidenceSpec(
        "AE04S",
        "Herrera Montano et al., Securecipher: an encryption system for insider-threat data leakage protection",
        "securecipher-insider-2024",
        "https://doi.org/10.1016/j.eswa.2024.124470",
        "2024-06-12",
        "2024-06-12",
        "ELSEVIER",
        "THREAT_INSIDER_THREAT",
        "PMT_CRYPTOGRAPHY",
        "SUPPORT",
        (
            "Securecipher is explicitly designed to use cryptographic protection against insider-threat data leakage. The evaluated "
            "prototype passed most of the reported NIST randomness tests and included context-based key generation and file marking, "
            "providing direct evidence that cryptography can be engineered for this threat model."
        ),
    ),
    ActionEvidenceSpec(
        "AE04C",
        "Herrera Montano et al., Securecipher: an encryption system for insider-threat data leakage protection",
        "securecipher-insider-2024",
        "https://doi.org/10.1016/j.eswa.2024.124470",
        "2024-06-12",
        "2024-06-12",
        "ELSEVIER",
        "THREAT_INSIDER_THREAT",
        "PMT_CRYPTOGRAPHY",
        "CHALLENGE",
        (
            "The insider-threat setting gives an authorized attacker stronger capabilities, including opportunities for known- or chosen-plaintext style attacks. "
            "The paper motivates a specialized construction rather than treating conventional encryption as automatically sufficient, so effectiveness depends on "
            "the cryptographic design and insider-access model."
        ),
    ),
    ActionEvidenceSpec(
        "AE05S",
        "Abdullah et al., Snort and Fail2ban as IDS for Brute Force Attack Mitigation",
        "snort-fail2ban-bruteforce-2024",
        "https://doi.org/10.26418/justin.v12i3.79617",
        "2024-07-31",
        "2024-07-31",
        "UNTAN_JUSTIN",
        "THREAT_BRUTE_FORCE_ATTACK",
        "PMT_IDS_IPS",
        "SUPPORT",
        (
            "A 2024 case study directly applies IDS components including Snort and Fail2Ban to brute-force defense, combined with a honeypot and alerting workflow. "
            "The study treats IDS/IPS-style monitoring and blocking as an implementable response to repeated authentication attacks rather than only a generic control."
        ),
    ),
    ActionEvidenceSpec(
        "AE05C",
        "Saito et al., TOPASE: Detection and Prevention of Brute Force Attacks with Disciplined IPs from IDS Logs",
        "topase-bruteforce-2016",
        "https://doi.org/10.2197/ipsjjip.24.217",
        "2016-03-15",
        "2016-03-15",
        "IPSJ",
        "THREAT_BRUTE_FORCE_ATTACK",
        "PMT_IDS_IPS",
        "CHALLENGE",
        (
            "Distributed low-rate brute-force attacks can stay below per-host login thresholds and evade ordinary IDS/IPS security rules. "
            "TOPASE was proposed because existing countermeasures were ineffective against this disciplined-IP pattern, documenting a direct limitation of conventional IDS/IPS use."
        ),
    ),
    ActionEvidenceSpec(
        "AE06S",
        "Zhang et al., ATT&CK-based APT risk propagation assessment model for zero trust networks",
        "apt-zero-trust-access-control-2024",
        "https://doi.org/10.1016/j.comnet.2024.110376",
        "2024-03-01",
        "2024-03-01",
        "ELSEVIER",
        "THREAT_TARGETED_ATTACK",
        "PMT_ACCESS_CONTROL",
        "SUPPORT",
        (
            "The study evaluates an ATT&CK-informed zero-trust risk model for advanced persistent threats and integrates the result with centralized access-control decisions. "
            "It reports improved ability to respond to sophisticated multi-stage threats, directly supporting adaptive access control as a targeted-attack defense mechanism."
        ),
    ),
    ActionEvidenceSpec(
        "AE06C",
        "Zhang et al., ATT&CK-based APT risk propagation assessment model for zero trust networks",
        "apt-zero-trust-access-control-2024",
        "https://doi.org/10.1016/j.comnet.2024.110376",
        "2024-03-01",
        "2024-03-01",
        "ELSEVIER",
        "THREAT_TARGETED_ATTACK",
        "PMT_ACCESS_CONTROL",
        "CHALLENGE",
        (
            "The same work explains that conventional perimeter and static access-control assumptions are insufficient for evolving APT behavior, and that existing risk assessment can miss dynamic attack progression. "
            "Effective access control therefore depends on continuously updated threat/risk context rather than the control category alone."
        ),
    ),
    ActionEvidenceSpec(
        "AE07S",
        "Cherckesova et al., The development of countermeasures against session hijacking",
        "session-hijacking-countermeasures-2024",
        "https://doi.org/10.1051/e3sconf/202453103019",
        "2024-06-01",
        "2024-06-01",
        "EDP_SCIENCES",
        "THREAT_SESSION_HIJACKING",
        "PMT_HTTPS",
        "SUPPORT",
        (
            "The 2024 session-hijacking study treats encrypted web communication and related secure-session practices as concrete countermeasures against interception-oriented hijacking. "
            "It supports HTTPS/TLS as an important transport-layer control within a broader session-protection strategy."
        ),
    ),
    ActionEvidenceSpec(
        "AE07C",
        "Yuasa et al., OIPM: Access Control Method to Prevent ID/Session Token Abuse on OpenID Connect",
        "oipm-session-token-2024",
        "https://doi.org/10.5220/0012757900003767",
        "2024-07-01",
        "2024-07-01",
        "SCITEPRESS",
        "THREAT_SESSION_HIJACKING",
        "PMT_HTTPS",
        "CHALLENGE",
        (
            "Session tokens can still be stolen through XSS or malicious browser extensions after a secure transport channel exists. "
            "The OIPM study notes that several existing session protections remain ineffective against user-level malware, showing that HTTPS alone cannot cover all session-hijacking paths."
        ),
    ),
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def action_evidence_registry_sha256() -> str:
    return hashlib.sha256(_canonical_json([spec.to_dict() for spec in ACTION_EVIDENCE_SPECS])).hexdigest()


def validate_action_evidence_registry() -> dict[str, Any]:
    cutoff = date.fromisoformat(ACTION_EVIDENCE_CUTOFF)
    actual_hash = action_evidence_registry_sha256()
    issues: list[str] = []
    ids = [spec.record_id for spec in ACTION_EVIDENCE_SPECS]
    if len(ids) != len(set(ids)):
        issues.append("action evidence record IDs must be unique")
    coverage: dict[tuple[str, str], set[str]] = {}
    for spec in ACTION_EVIDENCE_SPECS:
        if spec.direction not in {"SUPPORT", "CHALLENGE"}:
            issues.append(f"{spec.record_id} has invalid direction {spec.direction}")
        if date.fromisoformat(spec.available_at) > cutoff:
            issues.append(f"{spec.record_id} is post-cutoff ({spec.available_at})")
        coverage.setdefault((spec.threat_id, spec.pmt_id), set()).add(spec.direction)
    for pair, directions in sorted(coverage.items()):
        if directions != {"SUPPORT", "CHALLENGE"}:
            issues.append(f"{pair} lacks balanced action directions: {sorted(directions)}")
    if len(coverage) != 7:
        issues.append(f"action evidence registry must cover seven main Threat×PMT pairs, got {len(coverage)}")
    if ACTION_EVIDENCE_REGISTRY_SHA256 != "TO_BE_PINNED" and actual_hash != ACTION_EVIDENCE_REGISTRY_SHA256:
        issues.append(
            "action evidence registry changed without a new version/hash: "
            f"{actual_hash} != {ACTION_EVIDENCE_REGISTRY_SHA256}"
        )
    if issues:
        raise Stage0ValidationError("Action-evidence registry validation failed: " + "; ".join(issues))
    return {
        "registry_version": ACTION_EVIDENCE_REGISTRY_VERSION,
        "snapshot_id": ACTION_EVIDENCE_SNAPSHOT_ID,
        "cutoff": ACTION_EVIDENCE_CUTOFF,
        "registry_sha256": actual_hash,
        "pair_count": len(coverage),
    }


def load_action_evidence_records() -> tuple[list[EvidenceRecord], dict[str, Any]]:
    manifest = validate_action_evidence_registry()
    records: list[EvidenceRecord] = []
    for spec in ACTION_EVIDENCE_SPECS:
        content = " ".join(spec.summary.split())
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        claim_type = "action_effectiveness" if spec.direction == "SUPPORT" else "action_limitation"
        record = EvidenceRecord(
            evidence_id=f"{spec.record_id}_{content_hash[:16].upper()}",
            source_registry_id=spec.record_id,
            source_family="K",
            source=spec.source_name,
            publication_date=spec.publication_date,
            evidence_date=None,
            available_at=spec.available_at,
            source_document_id=spec.source_document_id,
            source_locator=f"{spec.source_uri}#curated-finding-{spec.direction.lower()}",
            source_snapshot_id=ACTION_EVIDENCE_SNAPSHOT_ID,
            evidence_chain_id=f"action-evidence:{spec.source_document_id}",
            evidence_type="curated_action_finding",
            retrieval_role="decision_action_curated",
            allowed_claim_types=(claim_type,),
            prohibited_claim_types=(
                "product_roi",
                "product_tco",
                "exact_budget",
                "exact_staffing_count",
                "universal_control_effectiveness",
            ),
            threat_ids=(spec.threat_id,),
            pmt_ids=(spec.pmt_id,),
            content=content,
            content_hash=content_hash,
        )
        record.validate()
        records.append(record)
    return sorted(records, key=lambda item: item.evidence_id), manifest


__all__ = [
    "ACTION_EVIDENCE_CUTOFF",
    "ACTION_EVIDENCE_REGISTRY_SHA256",
    "ACTION_EVIDENCE_REGISTRY_VERSION",
    "ACTION_EVIDENCE_SNAPSHOT_ID",
    "ACTION_EVIDENCE_SPECS",
    "action_evidence_registry_sha256",
    "load_action_evidence_records",
    "validate_action_evidence_registry",
]
