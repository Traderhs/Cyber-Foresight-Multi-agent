from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from Stage0.decision_evidence import DECISION_EVIDENCE_SLOT_CLAIM_TYPES
from Stage2.tests.test_stage2 import _result, _round
from Stage3.context_schema import ContextualDecisionObject, Stage3ContextualDecisionBundle
from Stage4.schema import DecisionLensEvaluation, LensAssessment, LensType, Stage4EvaluationBundle
from Stage4.tests.test_context_stage4 import _contextual_bundle, _evaluation_bundle
from Stage5.artifact import freeze_stage5_artifact, load_frozen_stage5_artifact
from Stage5.builder import build_stage5_diagnostic_bundle
from Stage5.schema import AgreementClass


def _hash(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _evidence_pack() -> dict:
    return {
        "case_id": "case__threat__pmt",
        "evidence": [
            {"evidence_id": "E1"},
            {"evidence_id": "E2"},
        ],
        "retrieval_metadata": {
            "required_evidence_slots": ["a", "b", "c"],
            "covered_evidence_slots": ["a", "b"],
            "missing_evidence_slots": ["c"],
            "source_family_count": 2,
            "independent_evidence_chain_count": 2,
            "temporal_violation_count": 0,
            "source_contract_violation_count": 0,
            "evidence_sufficiency": "PARTIAL",
        },
    }


def _grounded_assessment(obj_id: str, lens: LensType, stance: int) -> dict:
    if stance == 1:
        claims = [
            {
                "claim_id": f"{lens.value}-S",
                "statement": "Grounded support.",
                "evidence_ids": ["E3"],
                "status": "SUPPORTS_FEASIBILITY",
                "scope": "GENERAL",
            }
        ]
    elif stance == -1:
        claims = [
            {
                "claim_id": f"{lens.value}-C",
                "statement": "Grounded challenge.",
                "evidence_ids": ["E3"],
                "status": "CHALLENGES_FEASIBILITY",
                "scope": "GENERAL",
            }
        ]
    else:
        claims = [
            {
                "claim_id": f"{lens.value}-S",
                "statement": "Grounded support.",
                "evidence_ids": ["E3"],
                "status": "SUPPORTS_FEASIBILITY",
                "scope": "GENERAL",
            },
            {
                "claim_id": f"{lens.value}-C",
                "statement": "Grounded challenge.",
                "evidence_ids": ["E3"],
                "status": "CHALLENGES_FEASIBILITY",
                "scope": "GENERAL",
            },
        ]
    return {
        "decision_object_id": obj_id,
        "lens_type": lens.value,
        "stance": stance,
        "evidence_sufficiency": "PARTIAL",
        "rationale": "Deterministic Stage 5 fixture.",
        "claims": claims,
        "constraints": [],
        "evidence_gaps": [],
        "conditional_requirements": [],
    }


def _bundle_with_stances(stances: tuple[int, int, int]) -> dict:
    stage3 = _contextual_bundle()
    evaluations = []
    for raw in stage3["contextual_decision_objects"]:
        obj_id = raw["decision_object_id"]
        evaluations.append(
            DecisionLensEvaluation(
                decision_object_id=obj_id,
                technical_feasibility=_grounded_assessment(
                    obj_id, LensType.TECHNICAL_FEASIBILITY, stances[0]
                ),
                institutional_regional=_grounded_assessment(
                    obj_id, LensType.INSTITUTIONAL_REGIONAL, stances[1]
                ),
                financial_adoption=_grounded_assessment(
                    obj_id, LensType.FINANCIAL_ADOPTION, stances[2]
                ),
            )
        )
    return Stage4EvaluationBundle(
        case_id="case__threat__pmt",
        evaluations=evaluations,
    ).model_dump(mode="json")


def _add_eu_regulatory_context(stage3: dict, stage4: dict) -> tuple[dict, dict]:
    objects = []
    for raw in stage3["contextual_decision_objects"]:
        value = json.loads(json.dumps(raw))
        # Make the common E3 record institutionally direction-carrying for the fixture.
        value["decision_evidence_pack"]["slot_coverage"]["E3"] = ["governance_enablement"]
        covered = {"governance_enablement"}
        if value["context_scenario"]["jurisdiction_code"] == "EU":
            reg = {
                "evidence_id": "REG_EU",
                "content": "EU primary-law applicability context.",
                "source_registry_id": "31",
                "available_at": "2024-01-01",
                "source_snapshot_id": "snapshot-v1",
            }
            value["decision_evidence_ids"].append("REG_EU")
            value["evaluation_evidence_ids"].append("REG_EU")
            value["evaluation_evidence_records"].append(reg)
            value["decision_evidence_pack"]["decision_evidence_records"].append(reg)
            value["decision_evidence_pack"]["decision_evidence_ids"].append("REG_EU")
            value["decision_evidence_pack"]["slot_coverage"]["REG_EU"] = [
                "regulatory_applicability"
            ]
            covered.add("regulatory_applicability")
        value["decision_evidence_pack"]["retrieval_metadata"]["covered_slots"] = sorted(covered)
        value["decision_evidence_pack"]["retrieval_metadata"]["missing_slots"] = sorted(
            set(DECISION_EVIDENCE_SLOT_CLAIM_TYPES) - covered
        )
        objects.append(ContextualDecisionObject.model_validate(value))

    stage3_parsed = Stage3ContextualDecisionBundle(
        case_id="case__threat__pmt",
        base_stage3_input_fingerprint="f" * 64,
        scenario_set_version="stage3-context-scenarios-v1",
        contextual_decision_objects=objects,
    )

    evaluations = []
    by_id = {item.decision_object_id: item for item in objects}
    for raw_eval in Stage4EvaluationBundle.model_validate(stage4).evaluations:
        obj = by_id[raw_eval.decision_object_id]
        inst_data = raw_eval.institutional_regional.model_dump(mode="json")
        inst_data["stance"] = 1
        inst_data["evidence_sufficiency"] = "PARTIAL"
        inst_data["claims"] = [
            {
                "claim_id": "I-G",
                "statement": "General governance pathway exists.",
                "evidence_ids": ["E3"],
                "status": "SUPPORTS_FEASIBILITY",
                "scope": "GENERAL",
            }
        ]
        if obj.context_scenario.jurisdiction_code == "EU":
            inst_data["claims"].append(
                {
                    "claim_id": "I-EU",
                    "statement": "EU-specific deployment scope is grounded.",
                    "evidence_ids": ["E3", "REG_EU"],
                    "status": "SUPPORTS_FEASIBILITY",
                    "scope": "DEPLOYMENT_SPECIFIC",
                }
            )
        inst = LensAssessment.model_validate(inst_data)
        evaluations.append(
            DecisionLensEvaluation(
                decision_object_id=raw_eval.decision_object_id,
                technical_feasibility=raw_eval.technical_feasibility,
                institutional_regional=inst,
                financial_adoption=raw_eval.financial_adoption,
            )
        )
    return (
        stage3_parsed.model_dump(mode="json"),
        Stage4EvaluationBundle(
            case_id="case__threat__pmt",
            evaluations=evaluations,
        ).model_dump(mode="json"),
    )


class Stage5Tests(unittest.TestCase):
    def test_d_lens_and_grounded_agreement_classes(self):
        result = build_stage5_diagnostic_bundle(
            evidence_pack=_evidence_pack(),
            stage2_result=_result([_round(1)]),
            stage3_context_bundle=_contextual_bundle(),
            stage4_context_bundle=_bundle_with_stances((-1, 1, 0)),
        )
        for diagnostic in result.scenario_diagnostics:
            self.assertAlmostEqual(diagnostic.d_lens, 0.666667, places=6)
            self.assertTrue(diagnostic.d_lens_evaluable)
            self.assertEqual(diagnostic.directional_lens_count, 3)
            self.assertEqual(
                diagnostic.agreement_class,
                AgreementClass.SUBSTANTIVE_DISAGREEMENT,
            )

        agreement = build_stage5_diagnostic_bundle(
            evidence_pack=_evidence_pack(),
            stage2_result=_result([_round(1)]),
            stage3_context_bundle=_contextual_bundle(),
            stage4_context_bundle=_bundle_with_stances((1, 1, 1)),
        )
        self.assertTrue(
            all(
                item.agreement_class == AgreementClass.GROUNDED_AGREEMENT
                for item in agreement.scenario_diagnostics
            )
        )

    def test_joint_abstention_is_not_grounded_consensus(self):
        stage4 = _evaluation_bundle()
        result = build_stage5_diagnostic_bundle(
            evidence_pack=_evidence_pack(),
            stage2_result=_result([_round(1)]),
            stage3_context_bundle=_contextual_bundle(),
            stage4_context_bundle=stage4,
        )
        # Fixture has Technical PARTIAL/+1 and two insufficient lenses, so D exists
        # mathematically but is not a fully evaluable three-lens disagreement.
        for item in result.scenario_diagnostics:
            self.assertFalse(item.d_lens_evaluable)
            self.assertEqual(
                item.agreement_class,
                AgreementClass.EVIDENCE_LIMITED_DISAGREEMENT,
            )

        all_zero = _bundle_with_stances((0, 0, 0))
        for evaluation in all_zero["evaluations"]:
            for field in ("technical_feasibility", "institutional_regional", "financial_adoption"):
                assessment = evaluation[field]
                assessment["evidence_sufficiency"] = "INSUFFICIENT_EVIDENCE"
                assessment["claims"] = []
                assessment["evidence_gaps"] = ["No grounded directional evidence."]
        result = build_stage5_diagnostic_bundle(
            evidence_pack=_evidence_pack(),
            stage2_result=_result([_round(1)]),
            stage3_context_bundle=_contextual_bundle(),
            stage4_context_bundle=all_zero,
        )
        for item in result.scenario_diagnostics:
            self.assertTrue(item.joint_abstention)
            self.assertFalse(item.d_lens_evaluable)
            self.assertEqual(item.directional_lens_count, 0)
            self.assertEqual(
                item.agreement_class,
                AgreementClass.UNDER_INFORMED_CONVERGENCE,
            )

    def test_evidence_sensitivity_detects_same_stance_different_eu_grounding(self):
        stage3, stage4 = _add_eu_regulatory_context(
            _contextual_bundle(),
            _bundle_with_stances((1, 1, 0)),
        )
        result = build_stage5_diagnostic_bundle(
            evidence_pack=_evidence_pack(),
            stage2_result=_result([_round(1)]),
            stage3_context_bundle=stage3,
            stage4_context_bundle=stage4,
        )
        sensitivity = result.jurisdiction_sensitivity
        self.assertFalse(sensitivity.any_stance_sensitivity)
        self.assertTrue(sensitivity.any_evidence_sensitivity)
        self.assertEqual(sensitivity.evidence_sensitive_lenses, ["institutional_regional"])
        institutional = sensitivity.by_lens["institutional_regional"]
        self.assertFalse(institutional.jurisdiction_stance_sensitivity)
        self.assertTrue(institutional.jurisdiction_evidence_sensitivity)
        self.assertIn("REG_EU", institutional.jurisdiction_specific_available_evidence_ids_by_jurisdiction["EU"])
        self.assertIn("REG_EU", institutional.jurisdiction_specific_used_evidence_ids_by_jurisdiction["EU"])
        self.assertEqual(
            institutional.deployment_specific_directional_claim_count_by_jurisdiction,
            {"KR": 0, "EU": 1, "US": 0},
        )
        self.assertFalse(
            sensitivity.by_lens["technical_feasibility"].jurisdiction_evidence_sensitivity
        )
        self.assertFalse(
            sensitivity.by_lens["financial_adoption"].jurisdiction_evidence_sensitivity
        )

    def test_system_evidence_semantic_audit_is_not_faked(self):
        result = build_stage5_diagnostic_bundle(
            evidence_pack=_evidence_pack(),
            stage2_result=_result([_round(1)]),
            stage3_context_bundle=_contextual_bundle(),
            stage4_context_bundle=_bundle_with_stances((1, 1, 0)),
        )
        system = result.system_evidence_diagnostic
        self.assertAlmostEqual(system.evidence_slot_coverage_ratio, 2 / 3, places=6)
        self.assertEqual(system.claim_evidence_reference_pass_rate, 1.0)
        self.assertIsNone(system.claim_evidence_audit_pass_rate)
        self.assertEqual(
            system.claim_evidence_audit_status,
            "DEFERRED_TO_STAGE7_SEMANTIC_GROUNDING_AUDIT",
        )

    def test_artifact_round_trip_binds_exact_upstream_hashes(self):
        bundle = build_stage5_diagnostic_bundle(
            evidence_pack=_evidence_pack(),
            stage2_result=_result([_round(1)]),
            stage3_context_bundle=_contextual_bundle(),
            stage4_context_bundle=_bundle_with_stances((1, 1, 0)),
        )

        def upstream(case_id: str, output_key: str) -> dict:
            value = {"case_id": case_id, output_key: "o" * 64}
            value["artifact_sha256"] = _hash(value)
            return value

        stage2 = upstream("case__threat__pmt", "stage2_output_sha256")
        stage3 = upstream("case__threat__pmt", "stage3_contextual_output_sha256")
        stage4 = upstream("case__threat__pmt", "stage4_output_sha256")
        with tempfile.TemporaryDirectory() as tmp:
            artifact, path = freeze_stage5_artifact(
                project_root=tmp,
                evidence_pack=_evidence_pack(),
                stage2_artifact=stage2,
                stage3_context_artifact=stage3,
                stage4_context_artifact=stage4,
                diagnostic_bundle=bundle,
            )
            self.assertTrue(Path(path).exists())
            loaded, identity, loaded_path = load_frozen_stage5_artifact(
                project_root=tmp,
                evidence_pack=_evidence_pack(),
                stage2_artifact=stage2,
                stage3_context_artifact=stage3,
                stage4_context_artifact=stage4,
            )
            self.assertEqual(path, loaded_path)
            self.assertEqual(artifact["input_fingerprint"], identity["input_fingerprint"])
            self.assertEqual(loaded["stage5_output_sha256"], artifact["stage5_output_sha256"])


if __name__ == "__main__":
    unittest.main()
