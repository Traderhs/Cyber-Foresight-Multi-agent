from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


STAGE0_PARENT = Path(__file__).resolve().parents[2]
if str(STAGE0_PARENT) not in sys.path:
    sys.path.insert(0, str(STAGE0_PARENT))

from Stage0.builder import Stage0Builder  # noqa: E402
from Stage0.bm25 import BM25_RETRIEVER_VERSION, rank_bm25  # noqa: E402
from Stage0.evidence_store import EvidenceSnapshotWriter, EvidenceStore  # noqa: E402
from Stage0.paper_migration import PaperForecastMigrator  # noqa: E402
from Stage0.registry import SOURCE_REGISTRY, validate_source_registry  # noqa: E402
from Stage0.schema import (  # noqa: E402
    BasisType,
    ClaimReference,
    ClaimReferenceStatus,
    EvaluationMode,
    Stage0ValidationError,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Stage0RegistryTests(unittest.TestCase):
    def test_fixed_registry_contract(self) -> None:
        result = validate_source_registry()
        self.assertEqual(result["entry_count"], 32)
        self.assertEqual(result["family_count"], 9)
        self.assertEqual([source.registry_id for source in SOURCE_REGISTRY], [f"{i:02d}" for i in range(1, 33)])

    def test_held_out_sources_not_in_main_registry(self) -> None:
        names = " ".join(source.name.lower() for source in SOURCE_REGISTRY)
        self.assertNotIn("wef", names)
        self.assertNotIn("enisa", names)


class Stage0PaperMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tempdir = tempfile.TemporaryDirectory()
        cls.output_dir = Path(cls.tempdir.name) / "Forecast"
        cls.migrator = PaperForecastMigrator(PROJECT_ROOT, cls.output_dir)
        cls.migrator.run()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tempdir.cleanup()

    def test_paper_native_output_contract(self) -> None:
        result = self.migrator.validate_output()
        self.assertEqual(result["nodes"], 124)
        self.assertEqual(result["feature_dim"], 4)
        self.assertEqual(result["feature_slots"], 124 * 4)
        self.assertEqual(result["feature_rows"], 124 * 4 * 162)
        self.assertEqual(result["state_series"], 124)
        self.assertEqual(result["state_rows"], 124 * 198)

        manifest = json.loads((self.output_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], "stage0-paper-forecast-v3")
        self.assertEqual(
            manifest["derivation"],
            {
                "mode": "existing_experiment_artifacts_only",
                "model_training_executed": False,
                "model_inference_executed": False,
                "upstream_artifacts_are_read_only": True,
            },
        )
        self.assertEqual(manifest["paper_contract"]["logical_historical_shape"], [162, 124, 4])
        self.assertEqual(manifest["paper_contract"]["feature_names"], ["NoI", "NoP", "ACA", "PH"])
        self.assertEqual(manifest["paper_forecast_output_contract"]["status"], "PASS")
        self.assertNotIn("scalar_signal_count", manifest)

    def test_structured_masking_matches_available_modalities(self) -> None:
        manifest = json.loads((self.output_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["feature_availability_node_counts"],
            {"NoI": 16, "NoP": 98, "ACA": 124, "PH": 124},
        )
        contract = json.loads((self.output_dir / "feature_contract.json").read_text(encoding="utf-8"))
        pmt_noi = [slot for slot in contract if slot["node_type"] == "pmt" and slot["feature_name"] == "NoI"]
        threat_nop = [slot for slot in contract if slot["node_type"] == "threat" and slot["feature_name"] == "NoP"]
        self.assertEqual(len(pmt_noi), 98)
        self.assertEqual(len(threat_nop), 26)
        self.assertTrue(all(not slot["available"] and slot["mask_value"] == 0 for slot in pmt_noi))
        self.assertTrue(all(not slot["available"] and slot["mask_value"] == 0 for slot in threat_nop))

    def test_historical_x_is_separate_from_124_node_y(self) -> None:
        rows = (self.output_dir / "historical_node_features.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(rows), 124 * 4 * 162)
        self.assertNotIn("phase", json.loads(rows[0]))
        state_rows = [json.loads(line) for line in (self.output_dir / "node_state_series.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len([row for row in state_rows if row["phase"] == "forecast"]), 124 * 36)

    def test_graph_selects_exactly_124_node_state_series(self) -> None:
        contract = json.loads((self.output_dir / "node_state_contract.json").read_text(encoding="utf-8"))
        self.assertEqual(len(contract), 124)
        by_modality = {
            modality: sum(item["state_modality"] == modality for item in contract)
            for modality in {item["state_modality"] for item in contract}
        }
        self.assertEqual(by_modality, {"NoI": 16, "NoP": 108})
        ddos = next(item for item in contract if item["node_id"] == "THREAT_DDOS")
        dns = next(item for item in contract if item["node_id"] == "THREAT_DNS_SPOOFING")
        self.assertEqual(ddos["state_modality"], "NoI")
        self.assertEqual(dns["state_modality"], "NoP")

    def test_only_paper_124_node_state_series_are_validated(self) -> None:
        audit = json.loads((self.output_dir / "migration_audit.json").read_text(encoding="utf-8"))
        source = audit["source_match"]
        self.assertEqual(source["status"], "PASS")
        self.assertEqual(source["series_count"], 124)
        self.assertEqual(
            source["series_by_source_type"],
            {"incident": 16, "paper_threat": 10, "solution": 98},
        )
        self.assertEqual(source["historical_points_checked_per_series"], 162)
        self.assertLessEqual(source["max_abs_error"], source["tolerance"])
        manifest = json.loads((self.output_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["source_validation"]["node_state_series_count"], 124)

    def test_gap_table_is_derived_only_from_canonical_124_node_state(self) -> None:
        audit = json.loads((self.output_dir / "migration_audit.json").read_text(encoding="utf-8"))
        gaps = audit["gap_generation"]
        self.assertEqual(gaps["threat_count"], 26)
        self.assertEqual(gaps["gap_relation_count"], 303)
        self.assertEqual(gaps["gap_row_count"], 909)
        self.assertEqual(gaps["source"], "canonical paper-aligned 124-node state forecast")

    def test_available_historical_features_are_z_scored(self) -> None:
        rows = []
        with (self.output_dir / "historical_node_features.jsonl").open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row["node_id"] == "THREAT_DDOS" and row["feature_name"] == "NoI":
                    rows.append(row)
        values = [row["value_z"] for row in rows]
        self.assertEqual(len(values), 162)
        self.assertAlmostEqual(sum(values) / len(values), 0.0, places=10)
        variance = sum(value * value for value in values) / len(values)
        self.assertAlmostEqual(variance, 1.0, places=10)

    def test_selected_124_uncertainty_is_exposed_without_legacy_variance_relabel(self) -> None:
        manifest = json.loads((self.output_dir / "manifest.json").read_text(encoding="utf-8"))
        semantics = manifest["uncertainty_semantics"]
        self.assertIn("1.96", semantics["confidence_95_half_width"])
        self.assertEqual(semantics["paper_status"], "PAPER_124_NODE_Y")
        audit = json.loads((self.output_dir / "migration_audit.json").read_text(encoding="utf-8"))
        issue_ids = {item["id"] for item in audit["known_legacy_issues"]}
        self.assertEqual(issue_ids, {"LEGACY_STALE_BMTGNN_DATA", "LEGACY_VARIANCE_RESCALE"})


class Stage0EvidenceAndBuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.forecast_tempdir = tempfile.TemporaryDirectory()
        cls.forecast_dir = Path(cls.forecast_tempdir.name) / "Forecast"
        PaperForecastMigrator(PROJECT_ROOT, cls.forecast_dir).run()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.forecast_tempdir.cleanup()

    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.evidence_root = Path(self.tempdir.name) / "Evidence"
        writer = EvidenceSnapshotWriter(self.evidence_root, "test-snapshot-v1")
        self.pre_cutoff = writer.add_record(
            source_registry_id="02",
            source="Synthetic test extract from DBIR",
            publication_date="2024-05-01",
            available_at="2024-05-01",
            source_document_id="dbir-test",
            source_locator="p.1",
            evidence_type="observed_threat_reality",
            content="Observed breach data provides a threat-pattern evidence fixture for Stage 0 contract tests.",
            retrieval_role="observed_reality",
            threat_ids=("THREAT_DDOS",),
            pmt_ids=("PMT_NLP_LLM",),
        )
        writer.add_record(
            source_registry_id="02",
            source="Synthetic future test extract from DBIR",
            publication_date="2025-05-01",
            available_at="2025-05-01",
            source_document_id="dbir-future-test",
            source_locator="p.2",
            evidence_type="observed_threat_reality",
            content="This post-cutoff fixture must never enter a 2024 evidence pack.",
            retrieval_role="observed_reality",
            threat_ids=("THREAT_DDOS",),
            pmt_ids=("PMT_NLP_LLM",),
        )
        writer.add_record(
            source_registry_id="24",
            source="Synthetic untagged NIST fixture",
            publication_date="2024-02-26",
            available_at="2024-02-26",
            source_document_id="generic-guidance-test",
            source_locator="p.3",
            evidence_type="official_guidance_chunk",
            content="This generic guidance fixture has no explicit threat or PMT tag and must not be injected into every case.",
        )
        writer.finalize()
        self.snapshot = self.evidence_root / "snapshots/test-snapshot-v1"
        self.builder = Stage0Builder(self.forecast_dir, self.snapshot)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _pack(self):
        return self.builder.build(
            case_id="test-ddos-nlp",
            threat="DDoS",
            pmt="NLP/LLM",
            analysis_cutoff_date="2024-12-31",
            evaluation_mode=EvaluationMode.EX_ANTE_REPLAY,
            required_evidence_slots=("observed_threat_reality",),
        )

    def test_temporal_filter_excludes_future_evidence(self) -> None:
        pack = self._pack()
        self.assertEqual(len(pack.evidence), 1)
        self.assertEqual(pack.evidence[0].evidence_id, self.pre_cutoff.evidence_id)
        self.assertEqual(pack.retrieval_metadata["excluded_post_cutoff_record_count"], 1)
        self.assertEqual(pack.retrieval_metadata["temporal_violation_count"], 0)

    def test_forecast_specific_retrieval_metadata_is_explicit(self) -> None:
        pack = self._pack()
        query = pack.retrieval_metadata["query"]
        self.assertEqual(query["threat_id"], "THREAT_DDOS")
        self.assertEqual(query["pmt_id"], "PMT_NLP_LLM")
        self.assertEqual(query["forecast_horizon"], [2025, 2027])
        self.assertEqual(query["analysis_cutoff_date"], "2024-12-31")
        self.assertEqual(query["required_evidence_slots"], ["observed_threat_reality"])
        self.assertEqual(
            pack.retrieval_metadata["retrieval_intents"],
            ["support_candidate", "contradiction_candidate"],
        )
        self.assertEqual(
            pack.retrieval_metadata["semantic_stance_assignment"],
            "downstream_critic_not_stage0",
        )
        self.assertEqual(
            pack.retrieval_metadata["retrieval_method"],
            "hard_filter_then_okapi_bm25_then_source_diversity_top_k",
        )
        self.assertEqual(pack.retrieval_metadata["retriever_version"], BM25_RETRIEVER_VERSION)
        self.assertEqual(pack.retrieval_metadata["bm25_parameters"], {"k1": 1.2, "b": 0.75})
        self.assertIn("observed_threat_reality", pack.retrieval_metadata["retrieval_queries"])
        self.assertIn("DDoS", pack.retrieval_metadata["retrieval_queries"]["observed_threat_reality"])
        self.assertIn("NLP/LLM", pack.retrieval_metadata["retrieval_queries"]["observed_threat_reality"])

    def test_system_evidence_coverage_uses_semantic_slots(self) -> None:
        pack = self._pack()
        self.assertEqual(pack.retrieval_metadata["required_evidence_slots"], ["observed_threat_reality"])
        self.assertEqual(pack.retrieval_metadata["covered_evidence_slots"], ["observed_threat_reality"])
        self.assertEqual(pack.retrieval_metadata["missing_evidence_slots"], [])
        self.assertEqual(pack.retrieval_metadata["evidence_sufficiency"], "SUFFICIENT")
        self.assertEqual(
            pack.retrieval_metadata["evidence_slot_coverage"][self.pre_cutoff.evidence_id],
            ["observed_threat_reality"],
        )

    def test_untagged_evidence_is_not_treated_as_globally_relevant(self) -> None:
        pack = self._pack()
        self.assertTrue(all(record.threat_ids or record.pmt_ids for record in pack.evidence))

    def test_bm25_prefers_lexically_relevant_candidate_deterministically(self) -> None:
        documents = [
            "cloud compliance governance regional policy",
            "DDoS botnet amplification attack activity increased sharply",
            "generic security guidance and operations",
        ]
        query = "DDoS attack activity botnet amplification trend"
        first = rank_bm25(documents, query)
        second = rank_bm25(documents, query)
        self.assertEqual(first, second)
        self.assertEqual(first[0].index, 1)
        self.assertGreater(first[0].score, first[1].score)

    def test_builder_reads_paper_native_features_and_gap(self) -> None:
        pack = self._pack()
        self.assertEqual(pack.training_data_end, "2024-12-01")
        self.assertEqual(pack.forecast_origin_date, "2024-12-31")
        self.assertEqual(pack.forecast_summary["paper_feature_order"], ["NoI", "NoP", "ACA", "PH"])
        self.assertTrue(pack.forecast_summary["historical_input"]["threat_features"]["NoI"]["available"])
        self.assertFalse(pack.forecast_summary["historical_input"]["threat_features"]["NoP"]["available"])
        self.assertFalse(pack.forecast_summary["historical_input"]["pmt_features"]["NoI"]["available"])
        self.assertTrue(pack.forecast_summary["historical_input"]["pmt_features"]["NoP"]["available"])
        self.assertEqual(pack.forecast_summary["threat_state"]["state_modality"], "NoI")
        self.assertEqual(pack.forecast_summary["pmt_state"]["state_modality"], "NoP")
        self.assertEqual(set(pack.forecast_summary["gap_by_year"]), {"2025", "2026", "2027"})
        self.assertEqual(pack.forecast_summary["gap_semantics"], "yearly_mean_zscore_node_state_gap")
        self.assertEqual(pack.forecast_summary["predictive_uncertainty"]["status"], "PAPER_124_NODE_Y")
        self.assertEqual(len(pack.forecast_summary["forecast_field_ids"]["threat"]), 36)
        self.assertEqual(len(pack.forecast_summary["forecast_field_ids"]["pmt"]), 36)

    def test_ex_ante_cutoff_cannot_be_after_forecast_origin(self) -> None:
        with self.assertRaises(Stage0ValidationError):
            self.builder.build(
                case_id="bad-cutoff",
                threat="DDoS",
                pmt="NLP/LLM",
                analysis_cutoff_date="2025-01-01",
                evaluation_mode=EvaluationMode.EX_ANTE_REPLAY,
            )

    def test_valid_claim_reference(self) -> None:
        pack = self._pack()
        claim = ClaimReference(
            claim_id="C1",
            claim_type="observed_threat_pattern",
            basis_type=BasisType.EXTERNAL_EVIDENCE,
            evidence_ids=(self.pre_cutoff.evidence_id,),
        )
        audit = self.builder.audit_claim_reference(claim, pack)
        self.assertEqual(audit.status, ClaimReferenceStatus.VALID)

    def test_source_contract_blocks_roi_claim_from_dbir(self) -> None:
        pack = self._pack()
        claim = ClaimReference(
            claim_id="C2",
            claim_type="product_roi",
            basis_type=BasisType.EXTERNAL_EVIDENCE,
            evidence_ids=(self.pre_cutoff.evidence_id,),
        )
        audit = self.builder.audit_claim_reference(claim, pack)
        self.assertEqual(audit.status, ClaimReferenceStatus.SOURCE_CONTRACT_VIOLATION)

    def test_claim_without_any_reference_is_unsupported(self) -> None:
        audit = self.builder.audit_claim_reference(
            ClaimReference(claim_id="C3", claim_type="observed_threat_pattern", basis_type=BasisType.EXTERNAL_EVIDENCE),
            self._pack(),
        )
        self.assertEqual(audit.status, ClaimReferenceStatus.UNSUPPORTED_CLAIM)

    def test_evidence_snapshot_is_immutable(self) -> None:
        EvidenceStore(self.snapshot)
        with self.assertRaises(Stage0ValidationError):
            EvidenceSnapshotWriter(self.evidence_root, "test-snapshot-v1")

    def test_evidence_store_has_no_reasoning_insertion_api(self) -> None:
        store = EvidenceStore(self.snapshot)
        self.assertFalse(hasattr(store, "add_conversation_data"))
        self.assertFalse(hasattr(store, "add_reasoning"))

    def test_evidence_store_detects_raw_artifact_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir) / "Evidence"
            source = Path(tempdir) / "source.txt"
            source.write_text("frozen official evidence", encoding="utf-8")
            writer = EvidenceSnapshotWriter(root, "artifact-integrity-test")
            artifact = writer.add_artifact(
                source_registry_id="06",
                source_path=source,
                artifact_id="fixture",
                publication_date="2024-01-01",
                available_at="2024-01-01",
                version="fixture-v1",
            )
            snapshot = writer.finalize()
            EvidenceStore(snapshot)
            artifact_path = snapshot / artifact["snapshot_path"]
            artifact_path.write_bytes(b"tampered")
            with self.assertRaises(Stage0ValidationError):
                EvidenceStore(snapshot)


class Stage0LegacyBoundaryIntegrationTests(unittest.TestCase):
    def test_active_runtime_does_not_reopen_legacy_mutable_rag(self) -> None:
        main_source = (PROJECT_ROOT / "Multi-Agent/Pipeline/main.py").read_text(encoding="utf-8")
        stage0_nodes_source = (PROJECT_ROOT / "Multi-Agent/Stage0/nodes.py").read_text(encoding="utf-8")
        stage1_nodes_source = (PROJECT_ROOT / "Multi-Agent/Stage1/nodes.py").read_text(encoding="utf-8")
        nodes_source = stage0_nodes_source + stage1_nodes_source
        self.assertNotIn("cyber_rag", main_source)
        self.assertNotIn("cyber_rag", nodes_source)
        self.assertNotIn("get_rag_context", nodes_source)
        self.assertNotIn("add_conversation_data", nodes_source)
        self.assertIn("build_agent_input", nodes_source)


if __name__ == "__main__":
    unittest.main()
