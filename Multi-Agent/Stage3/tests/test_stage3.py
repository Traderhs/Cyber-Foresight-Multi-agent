from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from Stage3.artifact import build_stage3_identity, freeze_stage3_artifact, load_frozen_stage3_artifact
from Stage3.builder import build_stage3_bundle
from Stage3.nodes import _load_exact_stage2_artifact, stage3_build_node
from Stage3.schema import STAGE3_SELECTION_RULE_VERSION


def _evidence_pack() -> dict:
    return {
        "case_id": "case__threat__pmt",
        "source_snapshot_id": "snapshot-v1",
        "threat_id": "THREAT_X",
        "pmt_id": "PMT_X",
        "forecast_summary": {
            "threat_state": {"state_modality": "NoI", "direction": "increasing"},
            "pmt_state": {"state_modality": "NoP", "direction": "decreasing"},
            "gap_by_year": {"2025": 1.0, "2026": 2.0, "2027": 3.0},
            "gap_slope_per_year": 1.0,
            "gap_direction": "increasing",
            "gap_semantics": "threat_minus_pmt",
            "gap_direction_semantics": "positive_slope_means_widening",
            "predictive_uncertainty": {"status": "PAPER_124_NODE_Y"},
        },
        "evidence": [
            {"evidence_id": "E1", "pmt_ids": ["PMT_X"]},
            {"evidence_id": "E2", "pmt_ids": ["PMT_X", "PMT_ALT"]},
        ],
    }


def _claim(claim_id: str, relation: str, evidence_id: str) -> dict:
    return {
        "claim_id": claim_id,
        "statement": f"claim {claim_id}",
        "forecast_component": "THREAT_TRAJECTORY" if claim_id.startswith("A") else "PMT_TRAJECTORY",
        "forecast_relation": relation,
        "evidence_ids": [evidence_id],
    }


def _assessment(critic_type: str, claim: dict, stance: int) -> dict:
    decision = "SUPPORT_DOMINATES" if stance == 1 else "CHALLENGE_DOMINATES"
    return {
        "case_id": "case__threat__pmt",
        "critic_type": critic_type,
        "stance": stance,
        "evidence_sufficiency": "PARTIAL",
        "claims": [claim],
        "stance_basis": {
            "supporting_claim_ids": [claim["claim_id"]] if stance == 1 else [],
            "challenging_claim_ids": [claim["claim_id"]] if stance == -1 else [],
            "decision": decision,
            "rationale": "test stance basis",
        },
        "unresolved_questions": [],
        "unsupported_specificity_detected": False,
    }


def _stage2_result() -> dict:
    attack_pre = _assessment("attack_feasibility", _claim("A1", "SUPPORTS_FORECAST", "E1"), 1)
    defense_pre = _assessment("defense_robustness", _claim("D1", "CHALLENGES_FORECAST", "E2"), -1)
    adjudication = {
        "adjudication_id": "ADJ1",
        "attack_claim_ids": ["A1"],
        "defense_claim_ids": ["D1"],
        "outcome": "UNRESOLVED",
        "evidence_ids": ["E1", "E2"],
        "rationale": "Both interpretations remain materially supported.",
        "required_revision": None,
    }
    match = {
        "match_id": "M1",
        "attack_claim_ids": ["A1"],
        "defense_claim_ids": ["D1"],
        "status": "UNRESOLVED",
        "evidence_ids": ["E1", "E2"],
        "note": "same issue",
    }
    attack_turn = {
        "case_id": "case__threat__pmt",
        "round_index": 1,
        "responder_critic": "attack_feasibility",
        "target_critic": "defense_robustness",
        "responses": [
            {
                "target_claim_id": "D1",
                "response_type": "CHALLENGE",
                "evidence_ids": ["E1"],
                "note": "challenge",
            }
        ],
    }
    defense_turn = {
        "case_id": "case__threat__pmt",
        "round_index": 1,
        "responder_critic": "defense_robustness",
        "target_critic": "attack_feasibility",
        "responses": [
            {
                "target_claim_id": "A1",
                "response_type": "CHALLENGE",
                "evidence_ids": ["E2"],
                "note": "challenge",
            }
        ],
    }
    mediator = {
        "case_id": "case__threat__pmt",
        "round_index": 1,
        "claim_matches": [match],
        "evidence_conflicts": [],
        "evidence_gaps": [],
        "adjudications": [adjudication],
        "round_action": "STOP",
        "next_round_focus": [],
        "overall_rationale": "Stop after bounded adjudication.",
    }
    return {
        "case_id": "case__threat__pmt",
        "attack_pre_assessment": attack_pre,
        "defense_pre_assessment": defense_pre,
        "rounds": [
            {
                "round_index": 1,
                "attack_turn": attack_turn,
                "defense_turn": defense_turn,
                "mediator_summary": mediator,
            }
        ],
        "exchanges": [
            {
                "round_index": 1,
                "claim_id": "D1",
                "source_critic": "attack_feasibility",
                "target_critic": "defense_robustness",
                "response_type": "CHALLENGE",
                "evidence_ids": ["E1"],
                "note": "challenge",
            },
            {
                "round_index": 1,
                "claim_id": "A1",
                "source_critic": "defense_robustness",
                "target_critic": "attack_feasibility",
                "response_type": "CHALLENGE",
                "evidence_ids": ["E2"],
                "note": "challenge",
            },
        ],
        "resolved_claims": [],
        "unresolved_claims": [match],
        "evidence_conflicts": [],
        "evidence_gaps": [],
        "final_adjudications": [adjudication],
        "attack_post_assessment": attack_pre,
        "defense_post_assessment": defense_pre,
        "stance_changes": {"attack_feasibility": 0, "defense_robustness": 0},
    }


def _stage2_artifact(result: dict) -> dict:
    def h(value: object) -> str:
        raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()

    artifact = {
        "case_id": result["case_id"],
        "stage2_output_sha256": h(result),
        "debate_result": result,
    }
    artifact["artifact_sha256"] = h(artifact)
    return artifact


class Stage3BuilderTests(unittest.TestCase):
    def test_builder_freezes_current_case_pmt_only(self):
        bundle = build_stage3_bundle(
            evidence_pack=_evidence_pack(),
            stage0_config={"threat": "Threat X", "pmt": "PMT X"},
            stage2_result=_stage2_result(),
        )
        self.assertEqual(bundle.candidate_selection_rule, STAGE3_SELECTION_RULE_VERSION)
        self.assertEqual([item.action_id for item in bundle.eligible_action_set], ["PMT_X"])
        self.assertEqual(bundle.excluded_actions_and_reason[0].action_id, "PMT_ALT")
        self.assertEqual(len(bundle.decision_objects), 1)

    def test_decision_object_carries_stage2_and_forecast_provenance(self):
        result = _stage2_result()
        result["attack_post_assessment"]["unresolved_questions"] = ["attack unresolved"]
        result["defense_post_assessment"]["unresolved_questions"] = ["defense unresolved"]
        bundle = build_stage3_bundle(
            evidence_pack=_evidence_pack(),
            stage0_config={"threat": "Threat X", "pmt": "PMT X"},
            stage2_result=result,
        )
        obj = bundle.decision_objects[0]
        self.assertEqual(obj.stage2_critique_summary.attack_post_stance, 1)
        self.assertEqual(obj.stage2_critique_summary.defense_post_stance, -1)
        self.assertEqual(obj.stage2_critique_summary.matched_unresolved_claim_ids, ["A1", "D1"])
        self.assertEqual((obj.timing_window.start_year, obj.timing_window.end_year), (2025, 2027))
        self.assertEqual(obj.supporting_evidence_ids, ["E1"])
        self.assertEqual(obj.contradictory_evidence_ids, ["E2"])
        self.assertEqual(obj.evaluation_evidence_ids, ["E1", "E2"])
        self.assertIn("region", obj.unknown_context_fields)
        self.assertEqual(
            obj.stage2_critique_summary.attack_post_unresolved_questions,
            ["attack unresolved"],
        )
        self.assertEqual(
            obj.stage2_critique_summary.defense_post_unresolved_questions,
            ["defense unresolved"],
        )

    def test_decision_object_id_changes_with_deployment_context(self):
        pack = _evidence_pack()
        result = _stage2_result()
        generic = build_stage3_bundle(
            evidence_pack=pack,
            stage0_config={"threat": "Threat X", "pmt": "PMT X"},
            stage2_result=result,
        )
        regional = build_stage3_bundle(
            evidence_pack=pack,
            stage0_config={"threat": "Threat X", "pmt": "PMT X", "region": "EU"},
            stage2_result=result,
        )
        self.assertNotEqual(
            generic.decision_objects[0].decision_object_id,
            regional.decision_objects[0].decision_object_id,
        )

    def test_builder_rejects_stage0_config_that_conflicts_with_pack(self):
        with self.assertRaisesRegex(Exception, "Stage0 config mismatch"):
            build_stage3_bundle(
                evidence_pack=_evidence_pack(),
                stage0_config={"case_id": "wrong", "threat": "Threat X", "pmt": "PMT X"},
                stage2_result=_stage2_result(),
            )


class Stage3ArtifactTests(unittest.TestCase):
    def test_freeze_then_reuse_exact_stage3_artifact(self):
        pack = _evidence_pack()
        result = _stage2_result()
        stage2_artifact = _stage2_artifact(result)
        bundle = build_stage3_bundle(
            evidence_pack=pack,
            stage0_config={"threat": "Threat X", "pmt": "PMT X"},
            stage2_result=result,
        )
        config = {"threat": "Threat X", "pmt": "PMT X"}
        with tempfile.TemporaryDirectory() as tmp:
            artifact, path = freeze_stage3_artifact(
                project_root=tmp,
                evidence_pack=pack,
                stage2_artifact=stage2_artifact,
                stage0_config=config,
                decision_bundle=bundle,
            )
            loaded, _, loaded_path = load_frozen_stage3_artifact(
                project_root=tmp,
                evidence_pack=pack,
                stage2_artifact=stage2_artifact,
                stage0_config=config,
            )
            self.assertEqual(path, loaded_path)
            self.assertEqual(loaded["artifact_sha256"], artifact["artifact_sha256"])

    def test_stage3_identity_changes_when_decision_context_changes(self):
        pack = _evidence_pack()
        stage2_artifact = _stage2_artifact(_stage2_result())
        generic = build_stage3_identity(
            evidence_pack=pack,
            stage2_artifact=stage2_artifact,
            stage0_config={"threat": "Threat X", "pmt": "PMT X"},
        )
        eu = build_stage3_identity(
            evidence_pack=pack,
            stage2_artifact=stage2_artifact,
            stage0_config={"threat": "Threat X", "pmt": "PMT X", "region": "EU"},
        )
        self.assertNotEqual(generic["input_fingerprint"], eu["input_fingerprint"])
        self.assertEqual(eu["stage3_context"]["region"], "EU")

    def test_stage3_identity_rejects_tampered_stage2_artifact(self):
        pack = _evidence_pack()
        stage2_artifact = _stage2_artifact(_stage2_result())
        stage2_artifact["debate_result"]["case_id"] = "tampered"
        with self.assertRaisesRegex(Exception, "invalid content hash"):
            build_stage3_identity(
                evidence_pack=pack,
                stage2_artifact=stage2_artifact,
                stage0_config={"threat": "Threat X", "pmt": "PMT X"},
            )


class Stage3NodeTests(unittest.TestCase):
    def test_exact_stage2_loader_revalidates_stage1_and_stage2_chain(self):
        result = _stage2_result()
        stage1_artifact = {"artifact_sha256": "1" * 64}
        stage2_artifact = _stage2_artifact(result)
        state = {
            "forecast_data": "forecast",
            "evidence_pack": _evidence_pack(),
            "attack_assessment": result["attack_pre_assessment"],
            "defense_assessment": result["defense_pre_assessment"],
            "stage1_artifact_path": "stage1.json",
            "stage2_artifact_path": "stage2.json",
            "stage2_debate_result": result,
        }
        with (
            patch(
                "Stage3.nodes.load_frozen_stage1_artifact",
                return_value=(stage1_artifact, {}, Path("stage1.json").resolve()),
            ) as load_stage1,
            patch(
                "Stage3.nodes.load_frozen_stage2_artifact",
                return_value=(stage2_artifact, {}, Path("stage2.json").resolve()),
            ) as load_stage2,
        ):
            loaded = _load_exact_stage2_artifact(state)
        self.assertEqual(loaded, stage2_artifact)
        load_stage1.assert_called_once()
        load_stage2.assert_called_once()

    def test_exact_stage2_loader_rejects_state_path_mismatch(self):
        result = _stage2_result()
        state = {
            "forecast_data": "forecast",
            "evidence_pack": _evidence_pack(),
            "attack_assessment": result["attack_pre_assessment"],
            "defense_assessment": result["defense_pre_assessment"],
            "stage1_artifact_path": "stage1.json",
            "stage2_artifact_path": "wrong-stage2.json",
            "stage2_debate_result": result,
        }
        with (
            patch(
                "Stage3.nodes.load_frozen_stage1_artifact",
                return_value=({"artifact_sha256": "1" * 64}, {}, Path("stage1.json").resolve()),
            ),
            patch(
                "Stage3.nodes.load_frozen_stage2_artifact",
                return_value=(_stage2_artifact(result), {}, Path("stage2.json").resolve()),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "Stage 2 artifact path"):
                _load_exact_stage2_artifact(state)

    def test_stage3_node_reuses_frozen_artifact(self):
        result = _stage2_result()
        artifact = {"decision_bundle": {"decision_objects": [{"decision_object_id": "D1"}]}}
        state = {
            "stage2_complete": True,
            "stage2_artifact_path": "stage2.json",
            "stage2_debate_result": result,
            "evidence_pack": _evidence_pack(),
            "stage0_config": {"threat": "Threat X", "pmt": "PMT X"},
        }
        with (
            patch("Stage3.nodes._load_exact_stage2_artifact", return_value=_stage2_artifact(result)),
            patch(
                "Stage3.nodes.load_frozen_stage3_artifact",
                return_value=(artifact, {"input_fingerprint": "f" * 64}, Path("stage3.json")),
            ),
        ):
            output = stage3_build_node(state)
        self.assertTrue(output["stage3_complete"])
        self.assertTrue(output["stage3_artifact_reused"])
        self.assertEqual(output["decision_objects"], [{"decision_object_id": "D1"}])


if __name__ == "__main__":
    unittest.main()
