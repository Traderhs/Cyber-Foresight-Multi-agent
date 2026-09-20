from __future__ import annotations

from pathlib import Path
import unittest

from Stage7.audits import (
    extract_all_grounding_items,
    extract_all_mediator_items,
    select_manual_audit_sample,
    select_mediator_audit_sample,
)
from Stage7.config import (
    DECISION_ARCHITECTURE_VARIANTS,
    SENSITIVITY_CASE_IDS,
    experiment_manifest,
)
from Stage7.metrics import cluster_bootstrap_mean_ci
from Stage7.loaders import load_main_case_bundles
from Stage7.runner import run_stage7_validation
from Stage7.variant_generation import (
    FROZEN_A1_VARIANT_RUNNER_VERSION,
    STAGE7_DECISION_VARIANT_RUNNER_VERSION,
)
from Stage7.variants import load_variant_records


ROOT = Path(__file__).resolve().parents[3]


class Stage7HarnessTests(unittest.TestCase):
    def test_semantic_audit_samples_are_predeclared_and_representative(self) -> None:
        bundles = load_main_case_bundles(ROOT)
        grounding_population = extract_all_grounding_items(bundles)
        mediator_population = extract_all_mediator_items(bundles)
        grounding_sample = select_manual_audit_sample(grounding_population)
        mediator_sample = select_mediator_audit_sample(mediator_population)

        assert len(grounding_population) == 742
        assert len(grounding_sample) == 70
        assert len(mediator_population) == 51
        assert len(mediator_sample) == 23
        assert {item.case_id for item in mediator_sample} == {bundle.case_id for bundle in bundles}
        assert {item.outcome for item in mediator_sample} == {item.outcome for item in mediator_population}
        assert {item.audit_id for item in mediator_population if item.round_index > 1} <= {
            item.audit_id for item in mediator_sample
        }

    def test_final_manifest_targets_agent_layer_only(self) -> None:
        assert SENSITIVITY_CASE_IDS == (
            "main__ddos__nlp_llm",
            "main__malware__anomaly_detection",
            "main__session_hijacking__https",
        )
        assert DECISION_ARCHITECTURE_VARIANTS == (
            {
                "variant_id": "FULL_REDESIGNED_PIPELINE",
                "description": "Current contextual Stage 4 information-isolated three-lens evaluation with the frozen upstream Stage 1-3 chain and deterministic downstream policy held fixed.",
            },
            {
                "variant_id": "SINGLE_AGENT_BASELINE",
                "description": "Joint three-lens Stage 4 single-agent baseline over the same frozen contextual DecisionObjects/evidence and the same deterministic downstream policy.",
            },
        )
        manifest = experiment_manifest()
        assert [item["id"] for item in manifest["experiments"]] == ["A", "B", "C"]
        assert [item["name"] for item in manifest["experiments"]] == [
            "Robustness",
            "Decision architecture and cost",
            "Context contribution",
        ]
        assert set(manifest) == {
            "manifest_version",
            "harness_version",
            "statistical_contract_version",
            "prompt_paraphrase_variants",
            "sensitivity_case_ids",
            "decision_architecture_variants",
            "decision_policy_variants",
            "manual_audit_selection_seed",
            "manual_audit_per_stratum",
            "mediator_audit_selection_seed",
            "mediator_audit_per_case_outcome_stratum",
            "mediator_audit_include_all_followup_rounds",
            "experiments",
        }

    def test_case_cluster_bootstrap_resamples_cases_not_scenarios(self) -> None:
        values = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        clusters = ["case_a", "case_a", "case_a", "case_b", "case_b", "case_b"]
        ci = cluster_bootstrap_mean_ci(values, clusters, seed=7, repetitions=2000)
        assert ci is not None
        self.assertEqual(ci, (0.0, 1.0))

    def test_variant_runner_versions_preserve_completed_generation_contracts(self) -> None:
        assert FROZEN_A1_VARIANT_RUNNER_VERSION == "stage7-decision-variant-runner-v1"
        assert STAGE7_DECISION_VARIANT_RUNNER_VERSION == "stage7-decision-variant-runner-v3"

    def test_variant_inventory_uses_final_a_b_c_axis_names(self) -> None:
        variants = load_variant_records(ROOT)
        axes = {record.axis for record in variants}
        assert axes == {
            "A1_PROMPT_PARAPHRASE",
            "B_STAGE4_ARCHITECTURE_BASELINE",
            "C2_CONTEXTUALIZED_PATH_COMPARISON",
        }

    def test_deterministic_runner_freezes_three_axis_inventory(self) -> None:
        artifact, path = run_stage7_validation(ROOT, with_llm_audits=False)
        assert path.exists()
        assert artifact["main_case_count"] == 7
        assert artifact["main_scenario_count"] == 21
        statuses = {axis["experiment_id"]: axis["status"] for axis in artifact["axes"]}
        assert set(statuses) == {"A", "B", "C"}
        assert statuses["A"] == "EVALUATED"
        assert statuses["B"] == "PARTIAL"
        assert statuses["C"] == "EVALUATED"

        by_axis = {axis["experiment_id"]: axis for axis in artifact["axes"]}
        b_metrics = {item["name"]: item["value"] for item in by_axis["B"]["metrics"]}
        c_metrics = {item["name"]: item["value"] for item in by_axis["C"]["metrics"]}
        self.assertAlmostEqual(b_metrics["single_agent_baseline.recommendation_consistency_rate"], 2 / 7)
        self.assertAlmostEqual(c_metrics["contextualized_path.recommendation_consistency_rate"], 2 / 7)
        assert c_metrics["scenario_provenance.scenario_contract_violation_count"] == 0
        b_metric_records = {item["name"]: item for item in by_axis["B"]["metrics"]}
        c_metric_records = {item["name"]: item for item in by_axis["C"]["metrics"]}
        assert "case-cluster bootstrap" in b_metric_records[
            "single_agent_baseline.stance_agreement_mean"
        ]["notes"]
        assert "case-cluster bootstrap" in c_metric_records[
            "contextualized_path.stance_agreement_mean"
        ]["notes"]
        # Existing frozen semantic audit checkpoints are bound into the final
        # artifact without issuing any new LLM request.
        assert len(artifact["grounding_audits"]) in {0, 70}
        assert len(artifact["mediator_audits"]) in {0, 23}
        assert bool(artifact["grounding_audits"]) == bool(artifact["mediator_audits"])
        scenario_details = c_metrics["scenario_provenance.case_contract_details"]
        assert all(set(item["jurisdictions"]) == {"KR", "EU", "US"} for item in scenario_details)
        assert all(item["reference_profiles"] == ["CIS_IG2"] for item in scenario_details)


if __name__ == "__main__":
    unittest.main()
