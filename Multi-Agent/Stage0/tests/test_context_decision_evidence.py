from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from Stage0.context_scenarios import (
    CIS_IG2_SOURCE,
    CONTEXT_SCENARIO_SET_VERSION,
    cis_ig2_context_evidence_record,
    ensure_frozen_context_scenario_manifest,
    main_context_scenarios,
)
from Stage0.decision_evidence import build_decision_evidence_pack
from Stage0.decision_sources import (
    DECISION_SOURCE_MIN_DOCUMENTS_PER_PMT,
    DECISION_SOURCE_MIN_PUBLISHERS_PER_PMT,
    validate_decision_source_spec_coverage,
)
from Stage0.schema import EvidenceRecord


def _record(
    evidence_id: str,
    source_registry_id: str,
    allowed_claim_types: tuple[str, ...],
    content: str,
    *,
    available_at: str = "2024-01-01",
    pmt_ids: tuple[str, ...] = ("PMT_X",),
    retrieval_role: str = "neutral_candidate",
    source_document_id: str | None = None,
    source_snapshot_id: str = "snapshot-v1",
) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=evidence_id,
        source_registry_id=source_registry_id,
        source_family="G" if source_registry_id not in {"30", "31", "32"} else "I",
        source=f"source-{source_registry_id}",
        publication_date=available_at,
        evidence_date=available_at,
        available_at=available_at,
        source_document_id=source_document_id or f"doc-{evidence_id}",
        source_locator=f"loc-{evidence_id}",
        source_snapshot_id=source_snapshot_id,
        evidence_chain_id=evidence_id,
        evidence_type="test",
        retrieval_role=retrieval_role,
        allowed_claim_types=allowed_claim_types,
        prohibited_claim_types=(),
        threat_ids=("THREAT_X",),
        pmt_ids=pmt_ids,
        content=content,
        content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
    )


class _FakeStore:
    snapshot_id = "snapshot-v1"

    def __init__(self, records: list[EvidenceRecord]):
        self.records = records


class ContextScenarioTests(unittest.TestCase):
    def test_main_scenario_set_is_fixed_kr_eu_us_with_same_ig2_profile(self):
        scenarios = main_context_scenarios()
        self.assertEqual([item["scenario_id"] for item in scenarios], ["KR__CIS_IG2", "EU__CIS_IG2", "US__CIS_IG2"])
        self.assertTrue(all(item["scenario_set_version"] == CONTEXT_SCENARIO_SET_VERSION for item in scenarios))
        self.assertEqual(
            {item["context_selection_rule"] for item in scenarios},
            {"cis-ig2-fixed-enterprise-jurisdiction-sweep-v1"},
        )
        self.assertEqual({item["organization_type"] for item in scenarios}, {"CIS_IG2_REFERENCE_ENTERPRISE"})
        self.assertEqual(len({item["infrastructure_context"] for item in scenarios}), 1)
        self.assertEqual(len({item["budget_context"] for item in scenarios}), 1)

    def test_cis_context_record_is_cutoff_safe_and_non_numeric(self):
        record = cis_ig2_context_evidence_record()
        self.assertLessEqual(record["available_at"], "2024-12-31")
        self.assertIn("exact_budget", record["prohibited_claim_types"])
        self.assertIn("resource_profile", record["allowed_claim_types"])

    def test_cis_context_source_is_pinned_to_actual_ig2_document(self):
        self.assertEqual(
            CIS_IG2_SOURCE["source_uri"],
            "https://downloads.cisecurity.org/controls/CIS-Controls-Version-7-1.pdf",
        )
        self.assertEqual(
            CIS_IG2_SOURCE["source_content_sha256"],
            "ae55cb37c6e5ed108601fc159e9c5e1e66d68d0f599499472cbddd14ed2fd984",
        )
        self.assertEqual(CIS_IG2_SOURCE["source_locator"], "page 9, Implementation Group 2")
        self.assertLessEqual(
            CIS_IG2_SOURCE["methodological_reference"]["publication_date"],
            "2024-12-31",
        )
        self.assertNotIn("lotl-powershell", CIS_IG2_SOURCE["source_uri"])

    def test_context_scenario_manifest_is_frozen_and_rejects_in_place_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = ensure_frozen_context_scenario_manifest(project_root=tmp)
            self.assertTrue(path.exists())
            first = path.read_text(encoding="utf-8")
            self.assertEqual(path, ensure_frozen_context_scenario_manifest(project_root=tmp))
            payload = json.loads(first)
            payload["main_estimand"] = "mutated after freeze"
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "changed in place"):
                ensure_frozen_context_scenario_manifest(project_root=tmp)


class DecisionEvidenceTests(unittest.TestCase):
    def test_curated_decision_source_spec_has_at_least_three_documents_per_main_pmt(self):
        audit = validate_decision_source_spec_coverage()
        self.assertEqual(audit["minimum_documents_per_pmt"], DECISION_SOURCE_MIN_DOCUMENTS_PER_PMT)
        self.assertEqual(audit["minimum_publishers_per_pmt"], DECISION_SOURCE_MIN_PUBLISHERS_PER_PMT)
        self.assertTrue(
            all(
                len(documents) >= DECISION_SOURCE_MIN_DOCUMENTS_PER_PMT
                for documents in audit["document_coverage_by_pmt"].values()
            )
        )
        self.assertTrue(
            all(
                len(publishers) >= DECISION_SOURCE_MIN_PUBLISHERS_PER_PMT
                for publishers in audit["publisher_coverage_by_pmt"].values()
            )
        )

    def _store(self) -> _FakeStore:
        common_types = (
            "implementation_feasibility",
            "reference_architecture",
            "implementation_how_to",
            "organizational_applicability",
            "cyber_governance",
            "implementation_assessment",
        )
        return _FakeStore(
            [
                _record("E_IMPL", "27", common_types, "PMT X implementation integration staffing architecture requirement"),
                _record("E_CTRL", "24", ("cyber_governance", "organizational_applicability"), "PMT X governance resource adoption"),
                _record("E_KR", "30", ("legal_requirement", "regulatory_scope", "korea_control_requirement"), "Republic of Korea PMT X cybersecurity control requirement"),
                _record("E_EU", "31", ("legal_requirement", "regulatory_scope", "eu_legal_requirement"), "European Union PMT X cybersecurity control requirement"),
                _record("E_US", "32", ("legal_requirement", "regulatory_scope", "us_legal_requirement"), "United States PMT X cybersecurity control requirement"),
                _record("E_FUTURE", "27", common_types, "future implementation evidence", available_at="2025-01-01"),
            ]
        )

    def _pack(self, scenario: dict) -> dict:
        return build_decision_evidence_pack(
            store=self._store(),
            cutoff_date="2024-12-31",
            threat_id="THREAT_X",
            pmt_id="PMT_X",
            threat_name="Threat X",
            pmt_name="PMT X",
            scenario=scenario,
        )

    def test_generic_evidence_is_invariant_and_regulatory_evidence_is_jurisdiction_specific(self):
        packs = {item["jurisdiction_code"]: self._pack(item) for item in main_context_scenarios()}
        generic_sets = []
        for code, pack in packs.items():
            regulatory = {
                evidence_id
                for evidence_id, slots in pack["slot_coverage"].items()
                if "regulatory_applicability" in slots
            }
            generic_sets.append(set(pack["decision_evidence_ids"]) - regulatory)
            expected_regulatory = {"KR": "E_KR", "EU": "E_EU", "US": "E_US"}[code]
            self.assertIn(expected_regulatory, regulatory)
        self.assertTrue(all(item == generic_sets[0] for item in generic_sets[1:]))

    def test_post_cutoff_decision_evidence_is_excluded(self):
        pack = self._pack(main_context_scenarios()[0])
        self.assertNotIn("E_FUTURE", pack["decision_evidence_ids"])

    def test_semantic_gate_rejects_slot_word_overlap_without_pmt_relevance(self):
        common_types = ("implementation_feasibility", "implementation_how_to")
        store = _FakeStore(
            [
                _record(
                    "E_IRRELEVANT",
                    "27",
                    common_types,
                    "webMethods integration implementation configuration deployment requirement",
                    pmt_ids=(),
                )
            ]
        )
        pack = build_decision_evidence_pack(
            store=store,
            cutoff_date="2024-12-31",
            threat_id="THREAT_DDOS",
            pmt_id="PMT_NLP_LLM",
            threat_name="DDoS",
            pmt_name="NLP/LLM",
            scenario=main_context_scenarios()[0],
        )
        self.assertNotIn("E_IRRELEVANT", pack["decision_evidence_ids"])
        self.assertIn("technical_enablement", pack["retrieval_metadata"]["missing_slots"])

    def test_semantic_gate_accepts_predeclared_pmt_alias_in_untagged_record(self):
        common_types = ("implementation_feasibility", "implementation_how_to")
        store = _FakeStore(
            [
                _record(
                    "E_ALIAS",
                    "27",
                    common_types,
                    "A large language model can be implemented and deployed with integration and configuration controls.",
                    pmt_ids=(),
                )
            ]
        )
        pack = build_decision_evidence_pack(
            store=store,
            cutoff_date="2024-12-31",
            threat_id="THREAT_DDOS",
            pmt_id="PMT_NLP_LLM",
            threat_name="DDoS",
            pmt_name="NLP/LLM",
            scenario=main_context_scenarios()[0],
        )
        self.assertIn("E_ALIAS", pack["decision_evidence_ids"])

    def test_slot_proximity_gate_rejects_distant_unrelated_section(self):
        common_types = ("implementation_feasibility", "implementation_how_to")
        content = (
            "large language model research background "
            + ("unrelated narrative " * 80)
            + "implementation configuration deployment requirement for a different product"
        )
        store = _FakeStore(
            [_record("E_DISTANT", "27", common_types, content, pmt_ids=())]
        )
        pack = build_decision_evidence_pack(
            store=store,
            cutoff_date="2024-12-31",
            threat_id="THREAT_DDOS",
            pmt_id="PMT_NLP_LLM",
            threat_name="DDoS",
            pmt_name="NLP/LLM",
            scenario=main_context_scenarios()[0],
        )
        self.assertNotIn("E_DISTANT", pack["decision_evidence_ids"])

    def test_https_url_scheme_is_not_https_decision_evidence(self):
        common_types = ("implementation_feasibility", "implementation_how_to")
        store = _FakeStore(
            [
                _record(
                    "E_URL_ONLY",
                    "27",
                    common_types,
                    "Documentation is available at https://example.org and describes deployment configuration.",
                    pmt_ids=(),
                )
            ]
        )
        pack = build_decision_evidence_pack(
            store=store,
            cutoff_date="2024-12-31",
            threat_id="THREAT_SESSION_HIJACKING",
            pmt_id="PMT_HTTPS",
            threat_name="Session Hijacking",
            pmt_name="HTTPS",
            scenario=main_context_scenarios()[0],
        )
        self.assertNotIn("E_URL_ONLY", pack["decision_evidence_ids"])

    def test_operational_advisory_is_not_reused_as_generic_stage4_decision_evidence(self):
        store = _FakeStore(
            [
                _record(
                    "E_ADVISORY",
                    "10",
                    ("official_mitigation",),
                    "Intrusion detection deployment control recommendation.",
                    pmt_ids=("PMT_IDS_IPS",),
                )
            ]
        )
        pack = build_decision_evidence_pack(
            store=store,
            cutoff_date="2024-12-31",
            threat_id="THREAT_BRUTE_FORCE",
            pmt_id="PMT_IDS_IPS",
            threat_name="Brute Force Attack",
            pmt_name="IDS/IPS",
            scenario=main_context_scenarios()[0],
        )
        self.assertNotIn("E_ADVISORY", pack["decision_evidence_ids"])

    def test_cpg_non_action_chunk_is_excluded(self):
        store = _FakeStore(
            [
                _record(
                    "E_CPG_GLOSSARY",
                    "23",
                    ("recommended_control", "organizational_applicability"),
                    "Encryption is a cryptographic transformation. Implementation control definition.",
                    pmt_ids=("PMT_CRYPTOGRAPHY",),
                )
            ]
        )
        pack = build_decision_evidence_pack(
            store=store,
            cutoff_date="2024-12-31",
            threat_id="THREAT_RANSOMWARE",
            pmt_id="PMT_CRYPTOGRAPHY",
            threat_name="Ransomware",
            pmt_name="CRYPTOGRAPHY",
            scenario=main_context_scenarios()[0],
        )
        self.assertNotIn("E_CPG_GLOSSARY", pack["decision_evidence_ids"])

    def test_curated_decision_guidance_cannot_lexically_leak_to_another_pmt(self):
        record = _record(
            "DE_ACCESS",
            "DE11",
            ("implementation_feasibility", "implementation_how_to"),
            "Natural language processing may be discussed, but this access control deployment guidance is curated only for access control.",
            pmt_ids=("PMT_ACCESS_CONTROL",),
            retrieval_role="decision_guidance_curated",
            source_document_id="nist-sp-800-162",
            source_snapshot_id="decision-snapshot-v1",
        )
        pack = build_decision_evidence_pack(
            store=_FakeStore([]),
            cutoff_date="2024-12-31",
            threat_id="THREAT_DDOS",
            pmt_id="PMT_NLP_LLM",
            threat_name="DDoS",
            pmt_name="NLP/LLM",
            scenario=main_context_scenarios()[0],
            supplemental_records=(record,),
            supplemental_manifest={
                "snapshot_id": "decision-snapshot-v1",
                "manifest_sha256": "abc",
                "records_sha256": "def",
            },
        )
        self.assertNotIn("DE_ACCESS", pack["decision_evidence_ids"])

    def test_main_coverage_floor_rejects_fewer_than_three_distinct_generic_documents(self):
        records = tuple(
            _record(
                f"DE_{index}",
                f"DE{index:02d}",
                ("implementation_feasibility", "implementation_how_to"),
                f"Large language model deployment implementation configuration requirement {index}",
                pmt_ids=("PMT_NLP_LLM",),
                retrieval_role="decision_guidance_curated",
                source_document_id=f"doc-{index}",
                source_snapshot_id="decision-snapshot-v1",
            )
            for index in (1, 2)
        )
        with self.assertRaisesRegex(ValueError, "coverage floor failed"):
            build_decision_evidence_pack(
                store=_FakeStore([]),
                cutoff_date="2024-12-31",
                threat_id="THREAT_DDOS",
                pmt_id="PMT_NLP_LLM",
                threat_name="DDoS",
                pmt_name="NLP/LLM",
                scenario=main_context_scenarios()[0],
                supplemental_records=records,
                supplemental_manifest={
                    "snapshot_id": "decision-snapshot-v1",
                    "manifest_sha256": "abc",
                    "records_sha256": "def",
                },
                minimum_generic_records=3,
                minimum_generic_documents=3,
            )

    def test_main_coverage_floor_accepts_three_distinct_generic_documents(self):
        records = tuple(
            _record(
                f"DE_{index}",
                f"DE{index:02d}",
                (
                    "implementation_feasibility",
                    "implementation_limitation",
                    "deployment_maturity",
                    "adoption_benefit",
                    "adoption_barrier",
                    "resource_burden",
                    "governance_enablement",
                    "compliance_constraint",
                ),
                (
                    f"Large language model can be deployed and provides automation benefit {index}. "
                    "A production deployment is operational. A limitation creates complexity and risk. "
                    "Automation can reduce effort and improve efficiency, while staffing cost and training create burden. "
                    "Governance audit traceability enables management, while compliance requirements impose constraints."
                ),
                pmt_ids=("PMT_NLP_LLM",),
                retrieval_role="decision_guidance_curated",
                source_document_id=f"doc-{index}",
                source_snapshot_id="decision-snapshot-v1",
            )
            for index in (1, 2, 3)
        )
        pack = build_decision_evidence_pack(
            store=_FakeStore([]),
            cutoff_date="2024-12-31",
            threat_id="THREAT_DDOS",
            pmt_id="PMT_NLP_LLM",
            threat_name="DDoS",
            pmt_name="NLP/LLM",
            scenario=main_context_scenarios()[0],
            supplemental_records=records,
            supplemental_manifest={
                "snapshot_id": "decision-snapshot-v1",
                "manifest_sha256": "abc",
                "records_sha256": "def",
            },
            minimum_generic_records=3,
            minimum_generic_documents=3,
        )
        floor = pack["retrieval_metadata"]["coverage_floor"]
        self.assertGreaterEqual(floor["generic_record_count"], 3)
        self.assertEqual(floor["generic_document_count"], 3)

    def test_main_lens_floor_rejects_when_one_lens_has_fewer_than_three_documents(self):
        records = tuple(
            _record(
                f"DE_LENS_{index}",
                f"DL{index:02d}",
                ("implementation_feasibility", "implementation_how_to"),
                f"Large language model deployment implementation integration resource requirement {index}",
                pmt_ids=("PMT_NLP_LLM",),
                retrieval_role="decision_guidance_curated",
                source_document_id=f"lens-doc-{index}",
                source_snapshot_id="decision-snapshot-v1",
            )
            for index in (1, 2, 3)
        )
        with self.assertRaisesRegex(ValueError, "institutional_regional decision evidence documents="):
            build_decision_evidence_pack(
                store=_FakeStore([]),
                cutoff_date="2024-12-31",
                threat_id="THREAT_DDOS",
                pmt_id="PMT_NLP_LLM",
                threat_name="DDoS",
                pmt_name="NLP/LLM",
                scenario=main_context_scenarios()[0],
                supplemental_records=records,
                supplemental_manifest={
                    "snapshot_id": "decision-snapshot-v1",
                    "manifest_sha256": "abc",
                    "records_sha256": "def",
                },
                minimum_generic_records=3,
                minimum_generic_documents=3,
                minimum_lens_documents=3,
            )

    def test_balance_floor_rejects_burden_only_evidence(self):
        records = tuple(
            _record(
                f"DE_BURDEN_{index}",
                registry_id,
                ("adoption_barrier", "resource_burden", "implementation_limitation"),
                "Large language model staffing cost training maintenance burden and technical limitation risk.",
                pmt_ids=("PMT_NLP_LLM",),
                retrieval_role="decision_guidance_curated",
                source_document_id=f"burden-doc-{index}",
                source_snapshot_id="decision-snapshot-v1",
            )
            for index, registry_id in ((1, "DE01"), (2, "DE15"))
        )
        with self.assertRaisesRegex(ValueError, "financial_adoption.adoption_benefit"):
            build_decision_evidence_pack(
                store=_FakeStore([]),
                cutoff_date="2024-12-31",
                threat_id="THREAT_DDOS",
                pmt_id="PMT_NLP_LLM",
                threat_name="DDoS",
                pmt_name="NLP/LLM",
                scenario=main_context_scenarios()[0],
                supplemental_records=records,
                supplemental_manifest={
                    "snapshot_id": "decision-snapshot-v1",
                    "manifest_sha256": "abc",
                    "records_sha256": "def",
                },
                minimum_balance_documents_per_facet=2,
                minimum_balance_publishers_per_facet=2,
            )

    def test_balance_floor_accepts_two_sided_two_publisher_evidence(self):
        records = tuple(
            _record(
                f"DE_BAL_{index}",
                registry_id,
                (
                    "implementation_feasibility",
                    "implementation_limitation",
                    "deployment_maturity",
                    "adoption_benefit",
                    "adoption_barrier",
                    "resource_burden",
                    "governance_enablement",
                    "compliance_constraint",
                ),
                (
                    "Large language model can be deployed and provides automation that improves operations. "
                    "Production deployment and adoption are documented. Technical limitation, complexity and risk remain. "
                    "Automation can reduce effort and improve efficiency, while staffing cost, training and maintenance create burden. "
                    "Governance audit traceability supports accountability, while compliance requirements and privacy restrictions constrain use."
                ),
                pmt_ids=("PMT_NLP_LLM",),
                retrieval_role="decision_guidance_curated",
                source_document_id=f"balanced-doc-{index}",
                source_snapshot_id="decision-snapshot-v1",
            )
            for index, registry_id in ((1, "DE01"), (2, "DE15"), (3, "DE02"))
        )
        pack = build_decision_evidence_pack(
            store=_FakeStore([]),
            cutoff_date="2024-12-31",
            threat_id="THREAT_DDOS",
            pmt_id="PMT_NLP_LLM",
            threat_name="DDoS",
            pmt_name="NLP/LLM",
            scenario=main_context_scenarios()[0],
            supplemental_records=records,
            supplemental_manifest={
                "snapshot_id": "decision-snapshot-v1",
                "manifest_sha256": "abc",
                "records_sha256": "def",
            },
            minimum_balance_documents_per_facet=2,
            minimum_balance_publishers_per_facet=2,
            minimum_lens_publishers=2,
        )
        floor = pack["retrieval_metadata"]["coverage_floor"]
        for lens in floor["lens_coverage"].values():
            self.assertGreaterEqual(lens["publisher_count"], 2)
            for facet in lens["facet_coverage"].values():
                self.assertGreaterEqual(facet["document_count"], 2)
                self.assertGreaterEqual(facet["publisher_count"], 2)


if __name__ == "__main__":
    unittest.main()

