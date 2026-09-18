from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from .schema import Stage0ValidationError


SOURCE_REGISTRY_VERSION = "stage0-registry-v3"


@dataclass(frozen=True)
class SourceDefinition:
    registry_id: str
    family: str
    name: str
    ingestion_kind: str
    allowed_claim_types: tuple[str, ...]
    prohibited_claim_types: tuple[str, ...]
    official_domains: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        d = asdict(self)
        for key in ("allowed_claim_types", "prohibited_claim_types", "official_domains"):
            d[key] = list(d[key])
        return d


OBSERVED = ("observed_threat_pattern", "directional_threat_trend", "observed_attack_vector", "observed_adversary_behavior")
NO_FUTURE_OR_ROI = ("exact_future_probability", "product_roi", "product_tco", "universal_control_effectiveness")


SOURCE_REGISTRY: tuple[SourceDefinition, ...] = (
    SourceDefinition("01", "A", "Canonical B-MTGNN Forecast Dataset", "internal_canonical", ("forecast_trajectory", "threat_pmt_gap", "predictive_uncertainty", "forecast_provenance"), ("causal_effect", "real_world_effectiveness")),
    SourceDefinition("02", "B", "Verizon Data Breach Investigations Report (DBIR)", "annual_report", OBSERVED, NO_FUTURE_OR_ROI, ("verizon.com",)),
    SourceDefinition("03", "B", "Mandiant M-Trends", "annual_report", OBSERVED, NO_FUTURE_OR_ROI, ("cloud.google.com", "mandiant.com")),
    SourceDefinition("04", "B", "IBM X-Force Threat Intelligence Index", "annual_report", OBSERVED, NO_FUTURE_OR_ROI, ("ibm.com",)),
    SourceDefinition("05", "B", "Microsoft Digital Defense Report", "annual_report", OBSERVED + ("identity_cloud_ai_threat_trend",), NO_FUTURE_OR_ROI, ("microsoft.com",)),
    SourceDefinition("06", "C", "CVE Program / cvelistV5 Source Reference", "source_reference", ("cve_program_provenance", "cve_record_schema_reference"), ("vulnerability_identity", "affected_product_version", "confirmed_exploitation", "prevalence_magnitude"), ("cve.org", "github.com", "raw.githubusercontent.com")),
    SourceDefinition("07", "C", "NIST National Vulnerability Database (NVD)", "structured_feed", ("vulnerability_identity", "affected_product_version", "public_reference", "vulnerability_severity", "cwe_mapping", "cpe_mapping", "vulnerability_enrichment"), ("confirmed_exploitation", "business_impact"), ("nvd.nist.gov",)),
    SourceDefinition("08", "C", "CISA Known Exploited Vulnerabilities (KEV)", "structured_feed", ("confirmed_exploitation",), ("prevalence_magnitude", "vulnerability_severity"), ("cisa.gov",)),
    SourceDefinition("09", "C", "FIRST Exploit Prediction Scoring System (EPSS)", "structured_feed", ("exploitation_probability",), ("confirmed_exploitation", "business_impact"), ("first.org", "epss.cyentia.com")),
    SourceDefinition("10", "C", "CISA Cybersecurity Advisories / Malware Analysis Reports", "official_advisory", ("observed_ttp", "campaign_evidence", "malware_behavior", "official_mitigation"), ("product_roi",), ("cisa.gov", "github.com", "codeload.github.com", "raw.githubusercontent.com")),
    SourceDefinition("11", "C", "CERT/CC Vulnerability Notes Index", "source_reference", ("vulnerability_note_discovery", "certcc_source_reference"), ("vulnerability_mechanics", "affected_vendor", "official_remediation", "prevalence_magnitude", "product_roi"), ("kb.cert.org", "github.com")),
    SourceDefinition("12", "D", "MITRE ATT&CK", "knowledge_base", ("adversary_tactic", "adversary_technique", "adversary_procedure"), ("prevalence_magnitude", "causal_effectiveness"), ("attack.mitre.org", "raw.githubusercontent.com")),
    SourceDefinition("13", "D", "MITRE D3FEND", "knowledge_base", ("attack_defense_relation", "defensive_technique_mapping"), ("proven_effectiveness", "priority", "deployment_maturity"), ("d3fend.mitre.org",)),
    SourceDefinition("14", "D", "MITRE ATLAS", "knowledge_base", ("adversarial_ml_mechanism", "ai_attack_technique", "ai_case_mapping"), ("prevalence_magnitude", "causal_effectiveness"), ("atlas.mitre.org", "raw.githubusercontent.com")),
    SourceDefinition("15", "E", "NIST AI 100-2 Adversarial Machine Learning Taxonomy", "official_guidance", ("adversarial_ml_taxonomy", "ai_mitigation_class"), ("prevalence_magnitude", "deployment_effectiveness"), ("nist.gov",)),
    SourceDefinition("16", "E", "NIST AI 100-4 Synthetic Content Risk / Technical Approaches", "official_guidance", ("synthetic_content_risk", "deepfake_detection_class", "provenance_approach"), ("prevalence_magnitude", "deployment_effectiveness"), ("nist.gov",)),
    SourceDefinition("17", "E", "NSA/FBI/CISA Deepfake Threat Guidance", "official_guidance", ("deepfake_threat_mechanism", "deepfake_mitigation_guidance"), ("prevalence_magnitude", "exact_effectiveness"), ("cisa.gov", "nsa.gov", "fbi.gov", "media.defense.gov")),
    SourceDefinition("18", "E", "CISA Misinformation / Disinformation / Malinformation Guidance", "official_guidance", ("information_manipulation_mechanism", "critical_infrastructure_risk_framing"), ("prevalence_magnitude", "deployment_effectiveness"), ("cisa.gov",)),
    SourceDefinition("19", "F", "NISTIR 8259 Series IoT Device Cybersecurity", "official_guidance", ("iot_security_capability", "iot_manufacturer_support_baseline"), ("exact_effectiveness", "product_roi"), ("nist.gov",)),
    SourceDefinition("20", "F", "NIST SP 800-161 Rev.1 Cybersecurity Supply Chain Risk Management", "official_guidance", ("supply_chain_risk_control", "supply_chain_governance"), ("exact_effectiveness", "product_roi"), ("nist.gov",)),
    SourceDefinition("21", "F", "CISA Software Bill of Materials (SBOM) Resources", "official_guidance", ("software_dependency_transparency", "sbom_practice"), ("exact_effectiveness", "product_roi"), ("cisa.gov",)),
    SourceDefinition("22", "F", "CISA Insider Threat Mitigation Resources", "official_guidance", ("insider_threat_control", "insider_detection_mitigation"), ("exact_effectiveness", "product_roi"), ("cisa.gov",)),
    SourceDefinition("23", "G", "CISA Cybersecurity Performance Goals (CPG)", "official_guidance", ("recommended_control", "organizational_applicability", "prioritized_defensive_practice"), ("exact_effectiveness", "product_roi"), ("cisa.gov",)),
    SourceDefinition("24", "G", "NIST Cybersecurity Framework 2.0", "official_guidance", ("organizational_security_outcome", "cyber_governance"), ("exact_effectiveness", "product_roi"), ("nist.gov",)),
    SourceDefinition("25", "G", "NIST SP 800-53 Rev.5", "official_guidance", ("security_control", "privacy_control"), ("exact_effectiveness", "product_roi"), ("nist.gov",)),
    SourceDefinition("26", "G", "NIST SP 800-53A Rev.5", "official_guidance", ("control_assessment_procedure", "implementation_assessment"), ("exact_effectiveness", "product_roi"), ("nist.gov",)),
    SourceDefinition("27", "G", "NIST NCCoE SP 1800 Series", "reference_implementation", ("implementation_feasibility", "reference_architecture", "implementation_how_to"), ("universal_production_effectiveness", "product_roi"), ("nccoe.nist.gov", "nvlpubs.nist.gov")),
    SourceDefinition("28", "H", "arXiv", "academic_index", ("scholarly_work_discovery", "preprint_metadata", "academic_topic_metadata"), ("deployment_maturity", "operational_effectiveness"), ("arxiv.org", "export.arxiv.org")),
    SourceDefinition("29", "H", "Crossref", "academic_metadata", ("doi_verification", "publisher_metadata", "publication_metadata"), ("deployment_maturity", "operational_effectiveness"), ("crossref.org",)),
    SourceDefinition("30", "I", "Korea Primary Sources: law.go.kr + PIPC + KISA ISMS-P", "primary_law", ("legal_requirement", "regulatory_scope", "korea_control_requirement"), ("cross_jurisdiction_generalization",), ("law.go.kr", "pipc.go.kr", "kisa.or.kr", "isms-p.or.kr")),
    SourceDefinition("31", "I", "European Union Primary Source: EUR-Lex", "primary_law", ("legal_requirement", "regulatory_scope", "eu_legal_requirement"), ("cross_jurisdiction_generalization",), ("eur-lex.europa.eu",)),
    SourceDefinition("32", "I", "United States Primary Sources: eCFR + Federal Register + CISA Directives", "primary_law", ("legal_requirement", "regulatory_scope", "us_legal_requirement"), ("cross_jurisdiction_generalization",), ("ecfr.gov", "federalregister.gov", "cisa.gov")),
    SourceDefinition(
        "33",
        "H",
        "Crossref Cutoff-Constrained PMT Publication Trend",
        "academic_trend_aggregate",
        ("directional_publication_trend",),
        (
            "deployment_maturity",
            "operational_effectiveness",
            "causal_effectiveness",
        ),
        ("crossref.org",),
    ),
)


HELD_OUT_SOURCES = (
    "WEF Global Cybersecurity Outlook",
    "ENISA Foresight / Delphi-based expert assessment",
)


EXCLUDED_SOURCE_CLASSES = (
    "agent_generated_reasoning",
    "generic_web_search_result",
    "reddit_forum_social_media",
    "generic_vendor_marketing",
    "generic_vendor_pricing_roi",
    "exploit_poc_repository",
    "cloudflare_ddos_annual_report",
    "crowdstrike_annual_threat_report",
)


def registry_by_id() -> dict[str, SourceDefinition]:
    return {s.registry_id: s for s in SOURCE_REGISTRY}


def validate_source_registry(registry: Iterable[SourceDefinition] = SOURCE_REGISTRY) -> dict:
    items = tuple(registry)
    ids = [s.registry_id for s in items]
    expected = [f"{i:02d}" for i in range(1, 34)]
    families = {s.family for s in items}
    issues: list[str] = []
    if ids != expected:
        issues.append(f"registry IDs must be exactly 01..33 in order, got {ids}")
    if len(set(ids)) != len(ids):
        issues.append("registry IDs are not unique")
    if families != set("ABCDEFGHI"):
        issues.append(f"families must be A-I, got {sorted(families)}")
    for s in items:
        overlap = set(s.allowed_claim_types) & set(s.prohibited_claim_types)
        if overlap:
            issues.append(f"{s.registry_id} has allowed/prohibited overlap: {sorted(overlap)}")
        if not s.allowed_claim_types:
            issues.append(f"{s.registry_id} has no allowed claim types")
    names = " ".join(s.name.lower() for s in items)
    for held_out in ("wef", "enisa"):
        if held_out in names:
            issues.append(f"held-out source leaked into main registry: {held_out}")
    if issues:
        raise Stage0ValidationError("; ".join(issues))
    return {
        "registry_version": SOURCE_REGISTRY_VERSION,
        "entry_count": len(items),
        "family_count": len(families),
        "families": sorted(families),
        "held_out_sources": list(HELD_OUT_SOURCES),
        "excluded_source_classes": list(EXCLUDED_SOURCE_CLASSES),
    }
