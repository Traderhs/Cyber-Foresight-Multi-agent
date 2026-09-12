from __future__ import annotations

from dataclasses import dataclass


SOURCE_MANIFEST_VERSION = "stage0-source-manifest-v2"


@dataclass(frozen=True)
class ArtifactSpec:
    source_registry_id: str
    artifact_id: str
    acquisition: str  # local | url | arxiv_batch | crossref_batch
    locator: str
    parser: str
    evidence_type: str
    version: str
    publication_date: str | None = None
    available_at: str | None = None
    required: bool = True


# This manifest is intentionally closed.  Runtime retrieval may select records from
# these sources, but experiments must not add a new source outside this manifest.
ARTIFACT_SPECS: tuple[ArtifactSpec, ...] = (
    ArtifactSpec("02", "dbir-2023", "local", "Multi-Agent/Lagacy/inputs/__enqueued__/2023-data-breach-investigations-report-dbir.pdf", "pdf_chunks", "annual_report_chunk", "2023", "2023-12-31", "2023-12-31"),
    ArtifactSpec("02", "dbir-2024", "local", "Multi-Agent/Lagacy/inputs/__enqueued__/2024-dbir-data-breach-investigations-report.pdf", "pdf_chunks", "annual_report_chunk", "2024", "2024-12-31", "2024-12-31"),
    ArtifactSpec("02", "dbir-2025", "local", "Multi-Agent/Lagacy/inputs/__enqueued__/2025-dbir-data-breach-investigations-report.pdf", "pdf_chunks", "annual_report_chunk", "2025", "2025-12-31", "2025-12-31"),
    ArtifactSpec("03", "mtrends-2023", "local", "Multi-Agent/Lagacy/inputs/__enqueued__/m_trends_2023.pdf", "pdf_chunks", "annual_report_chunk", "2023", "2023-12-31", "2023-12-31"),
    ArtifactSpec("03", "mtrends-2024", "local", "Multi-Agent/Lagacy/inputs/__enqueued__/m-trends-2024.pdf", "pdf_chunks", "annual_report_chunk", "2024", "2024-12-31", "2024-12-31"),
    ArtifactSpec("03", "mtrends-2025", "local", "Multi-Agent/Lagacy/inputs/__enqueued__/m-trends-2025-en.pdf", "pdf_chunks", "annual_report_chunk", "2025", "2025-12-31", "2025-12-31"),
    ArtifactSpec("04", "xforce-2023", "local", "Multi-Agent/Lagacy/inputs/__enqueued__/IBM-Security-X-Force-Threat-Intelligence-Index-2023.pdf", "pdf_chunks", "annual_report_chunk", "2023", "2023-12-31", "2023-12-31"),
    ArtifactSpec("04", "xforce-2024", "local", "Multi-Agent/Lagacy/inputs/__enqueued__/IBM-XForce-Threat-Intelligence-Index-2024.pdf", "pdf_chunks", "annual_report_chunk", "2024", "2024-12-31", "2024-12-31"),
    ArtifactSpec("04", "xforce-2025", "local", "Multi-Agent/Lagacy/inputs/__enqueued__/ibm-x-force-threat-intelligence-index-2025-report.pdf", "pdf_chunks", "annual_report_chunk", "2025", "2025-12-31", "2025-12-31"),
    ArtifactSpec("05", "mddr-2025", "url", "https://www.microsoft.com/en-us/security/security-insider/threat-landscape/microsoft-digital-defense-report-2025", "html_chunks", "annual_report_chunk", "2025", "2025-10-16", "2025-10-16"),
    ArtifactSpec("06", "cve-list-v5-readme", "url", "https://raw.githubusercontent.com/CVEProject/cvelistV5/main/README.md", "text_chunks", "source_reference_chunk", "cvelistV5-frozen"),
    ArtifactSpec("07", "nvd-kev-cves", "url", "https://services.nvd.nist.gov/rest/json/cves/2.0?hasKev&resultsPerPage=2000", "nvd_json", "vulnerability_record", "snapshot"),
    ArtifactSpec("08", "cisa-kev", "url", "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json", "kev_json", "confirmed_exploitation_record", "snapshot"),
    ArtifactSpec("09", "epss-2023-12-31", "url", "https://epss.cyentia.com/epss_scores-2023-12-31.csv.gz", "epss_csv_gz", "exploitation_probability_record", "2023-12-31", "2023-12-31", "2023-12-31"),
    ArtifactSpec("09", "epss-2024-12-31", "url", "https://epss.cyentia.com/epss_scores-2024-12-31.csv.gz", "epss_csv_gz", "exploitation_probability_record", "2024-12-31", "2024-12-31", "2024-12-31"),
    ArtifactSpec("09", "epss-current", "url", "https://epss.cyentia.com/epss_scores-current.csv.gz", "epss_csv_gz", "exploitation_probability_record", "snapshot"),
    ArtifactSpec("10", "cisa-csaf", "url", "https://github.com/cisagov/CSAF/archive/refs/heads/develop.zip", "cisa_csaf_zip", "official_advisory_record", "develop-frozen"),
    ArtifactSpec("11", "cert-vulnerability-notes", "url", "https://kb.cert.org/vuls/", "html_chunks", "official_advisory_chunk", "snapshot"),
    ArtifactSpec("12", "attack-enterprise", "url", "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/enterprise-attack/enterprise-attack.json", "attack_stix", "attack_technique_record", "latest-frozen"),
    ArtifactSpec("13", "d3fend-ontology", "url", "https://d3fend.mitre.org/ontologies/d3fend.json", "d3fend_jsonld", "defensive_technique_record", "latest-frozen"),
    ArtifactSpec("14", "atlas-2026-08", "url", "https://raw.githubusercontent.com/mitre-atlas/atlas-data/main/dist/v6/ATLAS-2026.08.yaml", "atlas_yaml", "ai_attack_technique_record", "2026.08", "2026-08-01", "2026-08-01"),
    ArtifactSpec("15", "nist-ai-100-2-e2023", "url", "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-2e2023.pdf", "pdf_chunks", "official_guidance_chunk", "E2023", "2024-01-04", "2024-01-04"),
    ArtifactSpec("15", "nist-ai-100-2-e2025", "url", "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-2e2025.pdf", "pdf_chunks", "official_guidance_chunk", "E2025", "2025-03-24", "2025-03-24"),
    ArtifactSpec("16", "nist-ai-100-4", "url", "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-4.pdf", "pdf_chunks", "official_guidance_chunk", "100-4", "2024-11-20", "2024-11-20"),
    ArtifactSpec("17", "deepfake-guidance", "url", "https://media.defense.gov/2023/Sep/12/2003298925/-1/-1/0/CSI-DEEPFAKE-THREATS.PDF", "pdf_chunks", "official_guidance_chunk", "SEP-2023-v1.0", "2023-09-12", "2023-09-12"),
    ArtifactSpec("18", "cisa-mdm", "url", "https://www.cisa.gov/sites/default/files/2022-11/cisa_insight_mitigating_foreign_influence_508.pdf", "pdf_chunks", "official_guidance_chunk", "FEB-2022", "2022-02-01", "2022-02-01"),
    ArtifactSpec("19", "nistir-8259a", "url", "https://nvlpubs.nist.gov/nistpubs/ir/2020/NIST.IR.8259A.pdf", "pdf_chunks", "official_guidance_chunk", "8259A", "2020-05-29", "2020-05-29"),
    ArtifactSpec("19", "nistir-8259b", "url", "https://nvlpubs.nist.gov/nistpubs/ir/2021/NIST.IR.8259B.pdf", "pdf_chunks", "official_guidance_chunk", "8259B", "2021-08-25", "2021-08-25"),
    ArtifactSpec("20", "nist-sp-800-161r1", "url", "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-161r1.pdf", "pdf_chunks", "official_guidance_chunk", "800-161r1", "2022-05-05", "2022-05-05"),
    ArtifactSpec("21", "cisa-sbom-2024", "url", "https://www.cisa.gov/sites/default/files/2024-10/SBOM%20Framing%20Software%20Component%20Transparency%202024.pdf", "pdf_chunks", "official_guidance_chunk", "SBOM-THIRD-EDITION", "2024-09-03", "2024-09-03"),
    ArtifactSpec("22", "cisa-insider-threat", "url", "https://www.cisa.gov/sites/default/files/2022-11/Insider%20Threat%20Mitigation%20Guide_Final_508.pdf", "pdf_chunks", "official_guidance_chunk", "MITIGATION-GUIDE", "2020-11-01", "2020-11-01"),
    ArtifactSpec("23", "cisa-cpg", "url", "https://www.cisa.gov/sites/default/files/2023-03/CISA_CPG_REPORT_v1.0.1_FINAL.pdf", "pdf_chunks", "official_guidance_chunk", "1.0.1", "2023-03-01", "2023-03-01"),
    ArtifactSpec("24", "nist-csf-2", "url", "https://nvlpubs.nist.gov/nistpubs/CSWP/NIST.CSWP.29.pdf", "pdf_chunks", "official_guidance_chunk", "CSF-2.0", "2024-02-26", "2024-02-26"),
    ArtifactSpec("25", "nist-sp-800-53r5", "url", "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-53r5.pdf", "pdf_chunks", "official_guidance_chunk", "800-53r5", "2020-09-23", "2020-09-23"),
    ArtifactSpec("26", "nist-sp-800-53ar5", "url", "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-53Ar5.pdf", "pdf_chunks", "official_guidance_chunk", "800-53Ar5", "2022-01-25", "2022-01-25"),
    ArtifactSpec("27", "nist-sp-1800-35", "url", "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.1800-35.pdf", "pdf_chunks", "reference_implementation_chunk", "1800-35", "2025-06-10", "2025-06-10"),
    ArtifactSpec("28", "arxiv-security-batch", "arxiv_batch", "https://export.arxiv.org/api/query", "arxiv_atom", "academic_work", "snapshot"),
    ArtifactSpec("29", "crossref-pmt-batch", "crossref_batch", "https://api.crossref.org/works", "crossref_json", "academic_metadata_record", "snapshot"),
    ArtifactSpec("30", "korea-pipa-2020", "url", "https://www.law.go.kr/LSW/lsInfoP.do?chrClsCd=010203&lsiSeq=213857&urlMode=engLsInfoR&viewCls=engLsInfoR", "html_chunks", "primary_law_chunk", "PIPA-2020", "2020-02-04", "2020-02-04"),
    ArtifactSpec("30", "korea-pipa-2025", "url", "https://www.law.go.kr/lsInfoP.do?lsiSeq=270351&urlMode=engLsInfoR&viewCls=engLsInfoR", "html_chunks", "primary_law_chunk", "PIPA-2025", "2025-04-01", "2025-04-01"),
    ArtifactSpec("30", "korea-pipc", "url", "https://www.pipc.go.kr/eng/", "html_chunks", "primary_law_chunk", "snapshot"),
    ArtifactSpec("30", "korea-isms-p", "url", "https://www.isms-p.or.kr/sysm/intro/selectSysmCertDetail.do", "html_chunks", "primary_law_chunk", "snapshot"),
    ArtifactSpec("31", "eu-gdpr", "url", "https://eur-lex.europa.eu/eli/reg/2016/679/oj", "html_chunks", "primary_law_chunk", "GDPR", "2016-05-04", "2016-05-04"),
    ArtifactSpec("31", "eu-nis2", "url", "https://eur-lex.europa.eu/eli/dir/2022/2555/oj", "html_chunks", "primary_law_chunk", "NIS2", "2022-12-27", "2022-12-27"),
    ArtifactSpec("32", "us-cisa-bod-22-01", "url", "https://www.cisa.gov/sites/default/files/publications/Reducing_the_Significant_Risk_of_Known_Exploited_Vulnerabilities_20211103.pdf", "pdf_chunks", "primary_law_chunk", "BOD-22-01", "2021-11-03", "2021-11-03"),
    ArtifactSpec("32", "us-ecfr", "url", "https://www.ecfr.gov/api/search/v1/results?query=cybersecurity&per_page=100", "json_chunks", "primary_law_index_chunk", "snapshot"),
    ArtifactSpec("32", "us-federal-register", "url", "https://www.federalregister.gov/api/v1/documents.json?per_page=100&conditions%5Bterm%5D=cybersecurity", "json_chunks", "primary_law_index_chunk", "snapshot"),
)


def required_external_source_ids() -> tuple[str, ...]:
    return tuple(f"{i:02d}" for i in range(2, 33))

