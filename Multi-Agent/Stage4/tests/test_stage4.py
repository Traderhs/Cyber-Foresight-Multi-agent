from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from Pipeline.graph import create_context_free_graph, create_graph
from Stage3.schema import Stage3DecisionBundle
from Stage4.artifact import (
    build_stage4_identity,
    freeze_stage4_artifact,
    freeze_stage4_lens_checkpoint,
    load_frozen_stage4_artifact,
    load_stage4_lens_checkpoint,
)
from Stage4.input import build_stage4_evaluation_payload
from Stage4.nodes import _load_exact_stage3_artifact, prepare_stage4_node
from Stage4.schema import (
    DecisionLensEvaluation,
    LensAssessment,
    LensType,
    Stage4EvaluationBundle,
    build_constrained_lens_response_schema,
    validate_lens_assessment,
    validate_stage4_evaluation_bundle,
)


def _hash(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _evidence_pack() -> dict:
    return {
        "case_id": "case__threat__pmt",
        "source_snapshot_id": "snapshot-v1",
        "threat_id": "THREAT_X",
        "pmt_id": "PMT_X",
        "evidence": [
            {
                "evidence_id": "E1",
                "content": "Official guidance describes a generally applicable technical control.",
                "source_family": "official_guidance",
                "threat_ids": ["THREAT_X"],
                "pmt_ids": ["PMT_X"],
            },
            {
                "evidence_id": "E2",
                "content": "A second record describes implementation constraints without cost figures.",
                "source_family": "technical_report",
                "threat_ids": ["THREAT_X"],
                "pmt_ids": ["PMT_X"],
            },
        ],
    }


def _decision_object() -> dict:
    adjudication = {
        "adjudication_id": "ADJ1",
        "attack_claim_ids": ["A1"],
        "defense_claim_ids": ["D1"],
        "outcome": "ACCEPT_BOTH",
        "evidence_ids": ["E1", "E2"],
        "rationale": "Both bounded claims are supported.",
        "required_revision": None,
    }
    return {
        "decision_object_id": "decision__case__pmt__abc",
        "case_id": "case__threat__pmt",
        "threat": "Threat X",
        "threat_id": "THREAT_X",
        "pmt_or_mitigation": "PMT X",
        "pmt_or_mitigation_id": "PMT_X",
        "forecast_interpretation": {
            "threat_state": {"state_modality": "NoI", "direction": "increasing"},
            "pmt_state": {"state_modality": "NoP", "direction": "increasing"},
            "gap_by_year": {"2025": 1.0, "2026": 1.2, "2027": 1.4},
            "gap_slope_per_year": 0.2,
            "gap_direction": "increasing",
            "gap_semantics": "threat_minus_pmt",
            "gap_direction_semantics": "signed_gap_slope",
        },
        "stage2_critique_summary": {
            "attack_post_stance": 0,
            "defense_post_stance": 1,
            "final_mediator_adjudications": [adjudication],
            "matched_resolved_claim_ids": [],
            "matched_unresolved_claim_ids": [],
            "required_revisions": [],
            "attack_post_unresolved_questions": ["Need direct threat trajectory evidence."],
            "defense_post_unresolved_questions": [],
        },
        "candidate_action": {
            "action_id": "PMT_X",
            "action_name": "PMT X",
            "source": "B_MTGNN_CASE_PMT",
            "source_evidence_ids": ["E1", "E2"],
        },
        "candidate_action_source": "B_MTGNN_CASE_PMT",
        "candidate_selection_rule": "case-forecast-pmt-only-v1",
        "intended_goal": "Evaluate PMT X for Threat X.",
        "timing_window": {"start_year": 2025, "end_year": 2027},
        "deployment_context": {
            "region": None,
            "organization_type": None,
            "infrastructure_context": None,
            "budget_context": None,
        },
        "known_constraints": ["Do not infer missing deployment context."],
        "unknown_context_fields": [
            "region",
            "organization_type",
            "infrastructure_context",
            "budget_context",
        ],
        "evaluation_evidence_ids": ["E1", "E2"],
        "supporting_evidence_ids": ["E1"],
        "contradictory_evidence_ids": [],
        "unresolved_evidence_gaps": [],
        "predictive_uncertainty": {"method": "MC_dropout", "status": "available"},
    }


def _stage3_bundle() -> dict:
    obj = _decision_object()
    return Stage3DecisionBundle.model_validate(
        {
            "case_id": "case__threat__pmt",
            "candidate_selection_rule": "case-forecast-pmt-only-v1",
            "eligible_action_set": [obj["candidate_action"]],
            "excluded_actions_and_reason": [],
            "decision_objects": [obj],
        }
    ).model_dump(mode="json")


def _stage3_artifact() -> dict:
    bundle = _stage3_bundle()
    artifact = {
        "schema_version": "stage3-artifact-v2",
        "semantic_validation_version": "stage3-semantic-validation-v2",
        "candidate_selection_rule": "case-forecast-pmt-only-v1",
        "case_id": "case__threat__pmt",
        "input_fingerprint": "f" * 64,
        "stage0_evidence_pack_sha256": _hash(_evidence_pack()),
        "stage3_output_sha256": _hash(bundle),
        "decision_bundle": bundle,
    }
    artifact["artifact_sha256"] = _hash(artifact)
    return artifact


def _assessment(lens: LensType) -> dict:
    if lens == LensType.TECHNICAL_FEASIBILITY:
        return {
            "decision_object_id": "decision__case__pmt__abc",
            "lens_type": lens.value,
            "stance": 1,
            "evidence_sufficiency": "PARTIAL",
            "rationale": "The supplied records support general technical feasibility, but deployment context is absent.",
            "claims": [
                {
                    "claim_id": "T1",
                    "statement": "The supplied official guidance supports the general technical relevance of the control.",
                    "evidence_ids": ["E1"],
                    "status": "SUPPORTS_FEASIBILITY",
                    "scope": "GENERAL",
                }
            ],
            "constraints": ["Infrastructure-specific compatibility cannot be assessed."],
            "evidence_gaps": [],
            "conditional_requirements": ["Provide infrastructure context for deployment-specific assessment."],
        }
    return {
        "decision_object_id": "decision__case__pmt__abc",
        "lens_type": lens.value,
        "stance": 0,
        "evidence_sufficiency": "INSUFFICIENT_CONTEXT",
        "rationale": "The frozen deployment context is insufficient for a deployment-specific conclusion.",
        "claims": [],
        "constraints": [],
        "evidence_gaps": ["Required deployment context is not supplied."],
        "conditional_requirements": ["Supply the missing deployment context before a case-specific conclusion."],
    }


def _evaluation_bundle() -> dict:
    obj_id = _decision_object()["decision_object_id"]
    return Stage4EvaluationBundle(
        case_id="case__threat__pmt",
        evaluations=[
            DecisionLensEvaluation(
                decision_object_id=obj_id,
                technical_feasibility=_assessment(LensType.TECHNICAL_FEASIBILITY),
                institutional_regional=_assessment(LensType.INSTITUTIONAL_REGIONAL),
                financial_adoption=_assessment(LensType.FINANCIAL_ADOPTION),
            )
        ],
    ).model_dump(mode="json")


class Stage4SchemaTests(unittest.TestCase):
    def test_payload_is_exact_shared_evidence_universe(self):
        payload = build_stage4_evaluation_payload(
            decision_object=_decision_object(),
            evidence_pack=_evidence_pack(),
        )
        self.assertEqual(
            [item["evidence_id"] for item in payload["evaluation_evidence"]],
            ["E1", "E2"],
        )
        self.assertFalse(payload["evidence_boundary"]["sibling_lens_outputs_visible"])

    def test_constrained_schema_binds_object_lens_and_evidence_ids(self):
        schema = build_constrained_lens_response_schema(
            decision_object=_decision_object(),
            evidence_pack=_evidence_pack(),
            expected_lens_type=LensType.TECHNICAL_FEASIBILITY,
        )
        self.assertEqual(
            schema["properties"]["decision_object_id"]["const"],
            "decision__case__pmt__abc",
        )
        self.assertEqual(
            schema["properties"]["lens_type"]["const"],
            LensType.TECHNICAL_FEASIBILITY.value,
        )
        self.assertEqual(
            schema["$defs"]["LensClaim"]["properties"]["evidence_ids"]["items"]["enum"],
            ["E1", "E2"],
        )

    def test_missing_context_blocks_deployment_specific_claim(self):
        assessment = _assessment(LensType.TECHNICAL_FEASIBILITY)
        assessment["claims"][0]["scope"] = "DEPLOYMENT_SPECIFIC"
        with self.assertRaisesRegex(Exception, "DEPLOYMENT_SPECIFIC"):
            validate_lens_assessment(
                assessment,
                decision_object=_decision_object(),
                evidence_pack=_evidence_pack(),
                expected_lens_type=LensType.TECHNICAL_FEASIBILITY,
            )

    def test_financial_deployment_specific_claim_requires_infrastructure_context(self):
        obj = _decision_object()
        obj["deployment_context"] = {
            "region": None,
            "organization_type": "enterprise",
            "infrastructure_context": None,
            "budget_context": "bounded budget",
        }
        obj["unknown_context_fields"] = ["region", "infrastructure_context"]
        assessment = {
            "decision_object_id": "decision__case__pmt__abc",
            "lens_type": LensType.FINANCIAL_ADOPTION.value,
            "stance": 1,
            "evidence_sufficiency": "PARTIAL",
            "rationale": "Test deployment-specific financial claim.",
            "claims": [
                {
                    "claim_id": "F1",
                    "statement": "Deployment-specific adoption claim.",
                    "evidence_ids": ["E1"],
                    "status": "SUPPORTS_FEASIBILITY",
                    "scope": "DEPLOYMENT_SPECIFIC",
                }
            ],
            "constraints": [],
            "evidence_gaps": [],
            "conditional_requirements": [],
        }
        with self.assertRaisesRegex(Exception, "DEPLOYMENT_SPECIFIC"):
            validate_lens_assessment(
                assessment,
                decision_object=obj,
                evidence_pack=_evidence_pack(),
                expected_lens_type=LensType.FINANCIAL_ADOPTION,
            )

    def test_insufficient_context_cannot_emit_directional_stance(self):
        assessment = _assessment(LensType.INSTITUTIONAL_REGIONAL)
        assessment["stance"] = 1
        assessment["claims"] = [
            {
                "claim_id": "I1",
                "statement": "General claim",
                "evidence_ids": ["E1"],
                "status": "SUPPORTS_FEASIBILITY",
                "scope": "GENERAL",
            }
        ]
        with self.assertRaisesRegex(Exception, "requires stance=0"):
            LensAssessment.model_validate(assessment)

    def test_complete_bundle_validates_all_three_lenses(self):
        parsed = validate_stage4_evaluation_bundle(
            _evaluation_bundle(),
            decision_objects=[_decision_object()],
            evidence_pack=_evidence_pack(),
        )
        self.assertEqual(len(parsed.evaluations), 1)


class Stage4ArtifactTests(unittest.TestCase):
    def test_identity_binds_stage3_and_prompt_contract(self):
        identity = build_stage4_identity(
            evidence_pack=_evidence_pack(),
            stage3_artifact=_stage3_artifact(),
        )
        self.assertEqual(identity["case_id"], "case__threat__pmt")
        self.assertIn("technical_feasibility_system_prompt_sha256", identity["prompt_hashes"])
        self.assertIn("decision__case__pmt__abc", identity["decision_object_payload_sha256"])

    def test_checkpoint_freeze_and_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = freeze_stage4_lens_checkpoint(
                project_root=tmp,
                evidence_pack=_evidence_pack(),
                stage3_artifact=_stage3_artifact(),
                decision_object_id="decision__case__pmt__abc",
                lens_type=LensType.TECHNICAL_FEASIBILITY,
                assessment=_assessment(LensType.TECHNICAL_FEASIBILITY),
            )
            loaded, loaded_path = load_stage4_lens_checkpoint(
                project_root=tmp,
                evidence_pack=_evidence_pack(),
                stage3_artifact=_stage3_artifact(),
                decision_object_id="decision__case__pmt__abc",
                lens_type=LensType.TECHNICAL_FEASIBILITY,
            )
            self.assertEqual(path, loaded_path)
            self.assertEqual(loaded["assessment"]["stance"], 1)

    def test_final_artifact_freeze_and_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact, path = freeze_stage4_artifact(
                project_root=tmp,
                evidence_pack=_evidence_pack(),
                stage3_artifact=_stage3_artifact(),
                evaluation_bundle=_evaluation_bundle(),
            )
            loaded, _, loaded_path = load_frozen_stage4_artifact(
                project_root=tmp,
                evidence_pack=_evidence_pack(),
                stage3_artifact=_stage3_artifact(),
            )
            self.assertEqual(path, loaded_path)
            self.assertEqual(artifact["artifact_sha256"], loaded["artifact_sha256"])

    def test_tampered_stage3_artifact_is_rejected(self):
        artifact = _stage3_artifact()
        artifact["decision_bundle"]["case_id"] = "tampered"
        with self.assertRaisesRegex(Exception, "invalid content hash"):
            build_stage4_identity(
                evidence_pack=_evidence_pack(),
                stage3_artifact=artifact,
            )

    def test_self_consistent_stage3_artifact_from_other_evidence_pack_is_rejected(self):
        artifact = _stage3_artifact()
        artifact["stage0_evidence_pack_sha256"] = "0" * 64
        artifact["artifact_sha256"] = _hash(
            {key: value for key, value in artifact.items() if key != "artifact_sha256"}
        )
        with self.assertRaisesRegex(Exception, "current Stage 0 EvidencePack"):
            build_stage4_identity(
                evidence_pack=_evidence_pack(),
                stage3_artifact=artifact,
            )


class Stage4NodeTests(unittest.TestCase):
    def test_active_graph_runs_contextual_stage4_then_stage5_then_stage6(self):
        graph = create_graph().get_graph()
        edges = {(edge.source, edge.target) for edge in graph.edges}
        self.assertIn(("stage3_build", "stage3_contextualize"), edges)
        self.assertIn(("stage3_contextualize", "stage4_context_prepare"), edges)
        self.assertIn(("stage4_context_prepare", "contextual_priority_queue_lenses"), edges)
        self.assertIn(("contextual_priority_queue_lenses", "stage4_context_complete"), edges)
        self.assertIn(("stage4_context_complete", "stage5_diagnostics"), edges)
        self.assertIn(("stage5_diagnostics", "stage6_prepare"), edges)

    def test_context_free_graph_preserves_frozen_stage4_v1_path(self):
        graph = create_context_free_graph().get_graph()
        edges = {(edge.source, edge.target) for edge in graph.edges}
        self.assertIn(("stage3_build", "stage4_prepare"), edges)
        self.assertIn(("stage4_prepare", "technical_feasibility_lens"), edges)
        self.assertIn(("stage4_prepare", "institutional_regional_lens"), edges)
        self.assertIn(("stage4_prepare", "financial_adoption_lens"), edges)
        self.assertIn(("technical_feasibility_lens", "stage4_complete"), edges)
        self.assertIn(("institutional_regional_lens", "stage4_complete"), edges)
        self.assertIn(("financial_adoption_lens", "stage4_complete"), edges)
        self.assertIn(("stage4_complete", "__end__"), edges)

    def test_exact_stage3_loader_checks_state_path_and_bundle(self):
        state = {
            "stage3_complete": True,
            "evidence_pack": _evidence_pack(),
            "stage0_config": {"threat": "Threat X", "pmt": "PMT X"},
            "stage3_decision_bundle": _stage3_bundle(),
            "stage3_artifact_path": "stage3.json",
            "decision_objects": [_decision_object()],
        }
        with (
            patch("Stage4.nodes._load_exact_stage2_artifact", return_value={"artifact_sha256": "2" * 64}),
            patch(
                "Stage4.nodes.load_frozen_stage3_artifact",
                return_value=(_stage3_artifact(), {}, Path("stage3.json").resolve()),
            ),
        ):
            loaded = _load_exact_stage3_artifact(state)
        self.assertEqual(loaded["decision_bundle"], _stage3_bundle())

    def test_exact_stage3_loader_rejects_state_decision_object_mismatch(self):
        state = {
            "stage3_complete": True,
            "evidence_pack": _evidence_pack(),
            "stage0_config": {"threat": "Threat X", "pmt": "PMT X"},
            "stage3_decision_bundle": _stage3_bundle(),
            "stage3_artifact_path": "stage3.json",
            "decision_objects": [{**_decision_object(), "intended_goal": "mutated downstream state"}],
        }
        with (
            patch("Stage4.nodes._load_exact_stage2_artifact", return_value={"artifact_sha256": "2" * 64}),
            patch(
                "Stage4.nodes.load_frozen_stage3_artifact",
                return_value=(_stage3_artifact(), {}, Path("stage3.json").resolve()),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "state decision_objects"):
                _load_exact_stage3_artifact(state)

    def test_prepare_reuses_frozen_stage4_artifact_without_inference(self):
        frozen = {
            "evaluation_bundle": _evaluation_bundle(),
        }
        state = {
            "stage3_complete": True,
            "evidence_pack": _evidence_pack(),
            "stage3_decision_bundle": _stage3_bundle(),
            "stage3_artifact_path": "stage3.json",
            "decision_objects": [_decision_object()],
        }
        with (
            patch("Stage4.nodes._load_exact_stage3_artifact", return_value=_stage3_artifact()),
            patch(
                "Stage4.nodes.load_frozen_stage4_artifact",
                return_value=(frozen, {"input_fingerprint": "4" * 64}, Path("stage4.json")),
            ),
        ):
            output = prepare_stage4_node(state)
        self.assertTrue(output["stage4_artifact_reused"])
        self.assertEqual(len(output["technical_lens_assessments"]), 1)
        self.assertEqual(len(output["institutional_lens_assessments"]), 1)
        self.assertEqual(len(output["financial_lens_assessments"]), 1)


if __name__ == "__main__":
    unittest.main()

