from __future__ import annotations

import hashlib
import unittest

from Stage0.action_evidence import (
    ACTION_EVIDENCE_REGISTRY_SHA256,
    ACTION_EVIDENCE_REGISTRY_VERSION,
    load_action_evidence_records,
)
from Stage0.context_scenarios import main_context_scenarios
from Stage0.decision_evidence import build_decision_evidence_pack
from Stage0.main_guidance_selection import (
    MAIN_GUIDANCE_SELECTION_SHA256,
    MAIN_GUIDANCE_SELECTION_SPECS,
    MAIN_GUIDANCE_SELECTION_VERSION,
    load_main_guidance_selection,
    main_guidance_selection_sha256,
    validate_main_guidance_selection_spec,
)
from Stage0.schema import EvidenceRecord, Stage0ValidationError


def _record(
    evidence_id: str,
    *,
    pmt_id: str = "PMT_NLP_LLM",
    source_document_id: str | None = None,
    content: str = "audited guidance",
) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=evidence_id,
        source_registry_id="DE99",
        source_family="J",
        source="test-source",
        publication_date="2024-01-01",
        evidence_date=None,
        available_at="2024-01-01",
        source_document_id=source_document_id or f"doc-{evidence_id}",
        source_locator="test",
        source_snapshot_id="decision-snapshot-v1",
        evidence_chain_id=None,
        evidence_type="decision_guidance_chunk",
        retrieval_role="decision_guidance_curated",
        allowed_claim_types=("adoption_benefit", "resource_burden"),
        prohibited_claim_types=(),
        threat_ids=(),
        pmt_ids=(pmt_id,),
        content=content,
        content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
    )


class _FakeStore:
    snapshot_id = "snapshot-v1"

    def __init__(self, records: list[EvidenceRecord]):
        self.records = records


class V5MainEvidenceContractTests(unittest.TestCase):
    def test_main_guidance_selection_hash_and_six_pmt_coverage_are_pinned(self):
        audit = validate_main_guidance_selection_spec()
        self.assertEqual(audit["selection_version"], MAIN_GUIDANCE_SELECTION_VERSION)
        self.assertEqual(audit["selection_sha256"], MAIN_GUIDANCE_SELECTION_SHA256)
        self.assertEqual(main_guidance_selection_sha256(), MAIN_GUIDANCE_SELECTION_SHA256)
        self.assertEqual(audit["pmt_count"], 6)
        self.assertGreater(audit["record_count"], 0)

    def test_main_guidance_does_not_force_equal_direction_counts(self):
        access_specs = [
            spec for spec in MAIN_GUIDANCE_SELECTION_SPECS if spec.pmt_id == "PMT_ACCESS_CONTROL"
        ]
        benefit_count = sum("adoption_benefit" in spec.slots for spec in access_specs)
        burden_count = sum("adoption_burden" in spec.slots for spec in access_specs)
        self.assertEqual(benefit_count, 1)
        self.assertEqual(burden_count, 3)
        self.assertNotEqual(benefit_count, burden_count)

    def test_action_registry_has_support_and_challenge_for_all_seven_main_pairs(self):
        records, manifest = load_action_evidence_records()
        self.assertEqual(manifest["registry_version"], ACTION_EVIDENCE_REGISTRY_VERSION)
        self.assertEqual(manifest["registry_sha256"], ACTION_EVIDENCE_REGISTRY_SHA256)
        by_pair: dict[tuple[str, str], set[str]] = {}
        for record in records:
            pair = (record.threat_ids[0], record.pmt_ids[0])
            by_pair.setdefault(pair, set()).update(record.allowed_claim_types)
        self.assertEqual(len(by_pair), 7)
        for claim_types in by_pair.values():
            self.assertIn("action_effectiveness", claim_types)
            self.assertIn("action_limitation", claim_types)

    def test_exact_guidance_loader_rejects_missing_frozen_id(self):
        with self.assertRaisesRegex(Stage0ValidationError, "missing exact evidence_id"):
            load_main_guidance_selection(
                pmt_id="PMT_NLP_LLM",
                records=(),
            )

    def test_exact_main_selection_bypasses_bm25_for_generic_direction_slots(self):
        exact = _record("EXACT_BENEFIT", source_document_id="exact-doc")
        tempting = _record(
            "TEMPTING_BM25",
            source_document_id="tempting-doc",
            content=(
                "large language model adoption benefit reduce cost lower effort "
                "improve efficiency automation productivity"
            ),
        )
        pack = build_decision_evidence_pack(
            store=_FakeStore([tempting]),
            cutoff_date="2024-12-31",
            threat_id="THREAT_DDOS",
            pmt_id="PMT_NLP_LLM",
            threat_name="DDoS",
            pmt_name="NLP/LLM",
            scenario=main_context_scenarios()[0],
            main_guidance_records=(exact,),
            main_guidance_slot_coverage={"EXACT_BENEFIT": ("adoption_benefit",)},
            main_guidance_manifest={
                "selection_version": "test-selection-v1",
                "selection_sha256": "abc",
                "pmt_id": "PMT_NLP_LLM",
            },
        )
        self.assertIn("EXACT_BENEFIT", pack["decision_evidence_ids"])
        self.assertNotIn("TEMPTING_BM25", pack["decision_evidence_ids"])
        self.assertEqual(
            pack["retrieval_metadata"]["queries"]["adoption_benefit"],
            "EXACT_FROZEN_MAIN_GUIDANCE_SELECTION",
        )
        self.assertEqual(
            pack["retrieval_metadata"]["main_guidance_selection"]["selection_mode"],
            "exact_frozen_evidence_id_allowlist",
        )


if __name__ == "__main__":
    unittest.main()
