from __future__ import annotations

import re
import tempfile
import unittest

from Stage0.decision_evidence import (
    DECISION_EVIDENCE_RETRIEVAL_VERSION,
    DECISION_EVIDENCE_SLOT_CLAIM_TYPES,
)
from Stage3.context_artifact import freeze_stage3_context_artifact, load_frozen_stage3_context_artifact
from Stage3.context_schema import ContextualDecisionObject, Stage3ContextualDecisionBundle
from Stage4.context_artifact import (
    STAGE4_CONTEXT_GENERATION_CONTRACT_VERSION,
    build_stage4_context_identity,
    freeze_stage4_context_artifact,
    load_frozen_stage4_context_artifact,
)
from Stage4.context_nodes import (
    _build_jurisdiction_neutral_payload,
    _equivalence_synthetic_id,
)
from Stage4.context_input import build_context_stage4_evaluation_payload
from Stage4.context_prompts import CONTEXT_SYSTEM_PROMPT_BY_LENS
from Stage4.context_runtime import (
    STAGE4_CLIENT_CONCURRENCY,
    STAGE4_CONTEXT_RUNTIME_PROFILE_VERSION,
    STAGE4_CONTEXT_SIZE,
    STAGE4_PARALLEL_SLOTS,
    STAGE4_SERVER_CONTEXT_SIZE,
    get_stage4_context_runtime_profile,
)
from Stage4.context_schema import (
    build_constrained_context_lens_response_schema,
    context_lens_equivalence_key,
    validate_context_lens_assessment,
    validate_context_stage4_bundle,
)
from Stage4.schema import DecisionLensEvaluation, LensType, Stage4EvaluationBundle
from Stage4.tests.test_stage4 import _assessment, _decision_object, _evidence_pack, _stage3_artifact


def _context_record() -> dict:
    return {
        "evidence_id": "CTX1",
        "content": "CIS IG2 reference enterprise context.",
        "source_registry_id": "CTX_CIS_IG2",
        "available_at": "2019-03-30",
        "source_snapshot_id": "stage3-context-scenarios-v1",
    }


def _decision_record() -> dict:
    return {
        "evidence_id": "E3",
        "content": "Implementation guidance describes operational requirements.",
        "source_registry_id": "27",
        "available_at": "2024-01-01",
        "source_snapshot_id": "snapshot-v1",
    }


def _contextual_object(code: str, region: str, legal_source: str) -> dict:
    base = _decision_object()
    base["decision_object_id"] = f"contextual__decision__{code.lower()}"
    base["deployment_context"] = {
        "region": region,
        "organization_type": "CIS_IG2_REFERENCE_ENTERPRISE",
        "infrastructure_context": "Multi-department enterprise IT.",
        "budget_context": "No numeric budget is assumed.",
    }
    base["unknown_context_fields"] = []
    base["evaluation_evidence_ids"] = ["E1", "E2", "CTX1", "E3"]
    base.update(
        {
            "base_decision_object_id": "decision__case__pmt__abc",
            "scenario_id": f"{code}__CIS_IG2",
            "scenario_set_version": "stage3-context-scenarios-v1",
            "context_scenario": {
                "scenario_id": f"{code}__CIS_IG2",
                "scenario_set_version": "stage3-context-scenarios-v1",
                "context_selection_rule": "cis-ig2-fixed-enterprise-jurisdiction-sweep-v1",
                "jurisdiction_code": code,
                "jurisdiction_source_registry_id": legal_source,
                "region": {
                    "value": region,
                    "provenance_type": "SCENARIO_PARAMETER",
                    "source_evidence_ids": [],
                },
                "organization_type": {
                    "value": "CIS_IG2_REFERENCE_ENTERPRISE",
                    "provenance_type": "STANDARD_DERIVED",
                    "source_evidence_ids": ["CTX1"],
                },
                "infrastructure_context": {
                    "value": "Multi-department enterprise IT.",
                    "provenance_type": "STANDARD_DERIVED",
                    "source_evidence_ids": ["CTX1"],
                },
                "budget_context": {
                    "value": "No numeric budget is assumed.",
                    "provenance_type": "SCENARIO_PARAMETER",
                    "source_evidence_ids": [],
                },
                "reference_enterprise_profile": {
                    "value": "CIS_IG2",
                    "provenance_type": "STANDARD_DERIVED",
                    "source_evidence_ids": ["CTX1"],
                },
                "assumptions": ["Reference enterprise fixed before Stage 4."],
                "non_assumptions": ["No numeric budget is assumed."],
            },
            "forecast_evidence_ids": ["E1", "E2"],
            "context_evidence_ids": ["CTX1"],
            "decision_evidence_ids": ["E3"],
            "evaluation_evidence_records": [
                *_evidence_pack()["evidence"],
                _context_record(),
                _decision_record(),
            ],
            "decision_evidence_pack": {
                "retrieval_version": DECISION_EVIDENCE_RETRIEVAL_VERSION,
                "scenario_id": f"{code}__CIS_IG2",
                "source_snapshot_id": "snapshot-v1",
                "analysis_cutoff_date": "2024-12-31",
                "context_evidence_records": [_context_record()],
                "decision_evidence_records": [_decision_record()],
                "context_evidence_ids": ["CTX1"],
                "decision_evidence_ids": ["E3"],
                "slot_coverage": {"E3": ["technical_enablement"]},
                "retrieval_metadata": {
                    "required_slots": list(DECISION_EVIDENCE_SLOT_CLAIM_TYPES),
                    "covered_slots": ["technical_enablement"],
                    "missing_slots": sorted(
                        set(DECISION_EVIDENCE_SLOT_CLAIM_TYPES) - {"technical_enablement"}
                    ),
                },
            },
        }
    )
    return ContextualDecisionObject.model_validate(base).model_dump(mode="json")


def _contextual_object_with_decision_slot(slot: str) -> ContextualDecisionObject:
    value = _contextual_object("KR", "Republic of Korea", "30")
    value["decision_evidence_pack"]["slot_coverage"] = {"E3": [slot]}
    value["decision_evidence_pack"]["retrieval_metadata"]["covered_slots"] = [slot]
    value["decision_evidence_pack"]["retrieval_metadata"]["missing_slots"] = sorted(
        set(DECISION_EVIDENCE_SLOT_CLAIM_TYPES) - {slot}
    )
    return ContextualDecisionObject.model_validate(value)


def _contextual_object_with_decision_slots(
    slots_by_evidence_id: dict[str, list[str]],
) -> ContextualDecisionObject:
    value = _contextual_object("KR", "Republic of Korea", "30")
    decision_records = []
    for evidence_id in slots_by_evidence_id:
        record = _decision_record()
        record["evidence_id"] = evidence_id
        record["content"] = f"Decision guidance for {evidence_id}."
        decision_records.append(record)
    decision_ids = list(slots_by_evidence_id)
    value["decision_evidence_ids"] = decision_ids
    value["evaluation_evidence_ids"] = ["E1", "E2", "CTX1", *decision_ids]
    value["evaluation_evidence_records"] = [
        *_evidence_pack()["evidence"],
        _context_record(),
        *decision_records,
    ]
    value["decision_evidence_pack"]["decision_evidence_records"] = decision_records
    value["decision_evidence_pack"]["decision_evidence_ids"] = decision_ids
    value["decision_evidence_pack"]["slot_coverage"] = slots_by_evidence_id
    covered_slots = sorted({slot for slots in slots_by_evidence_id.values() for slot in slots})
    value["decision_evidence_pack"]["retrieval_metadata"]["covered_slots"] = covered_slots
    value["decision_evidence_pack"]["retrieval_metadata"]["missing_slots"] = sorted(
        set(DECISION_EVIDENCE_SLOT_CLAIM_TYPES) - set(covered_slots)
    )
    return ContextualDecisionObject.model_validate(value)


def _contextual_bundle() -> dict:
    objects = [
        _contextual_object("KR", "Republic of Korea", "30"),
        _contextual_object("EU", "European Union", "31"),
        _contextual_object("US", "United States", "32"),
    ]
    return Stage3ContextualDecisionBundle(
        case_id="case__threat__pmt",
        base_stage3_input_fingerprint="f" * 64,
        scenario_set_version="stage3-context-scenarios-v1",
        contextual_decision_objects=objects,
    ).model_dump(mode="json")


def _assessment_for(obj_id: str, lens: LensType) -> dict:
    value = _assessment(lens)
    value["decision_object_id"] = obj_id
    if lens == LensType.TECHNICAL_FEASIBILITY and value.get("claims"):
        # Contextual v4 directional feasibility must be grounded in the
        # decision-evidence supplement, not forecast evidence alone.
        value["claims"][0]["evidence_ids"] = ["E3"]
    return value


def _evaluation_bundle() -> dict:
    evaluations = []
    for raw in _contextual_bundle()["contextual_decision_objects"]:
        obj_id = raw["decision_object_id"]
        evaluations.append(
            DecisionLensEvaluation(
                decision_object_id=obj_id,
                technical_feasibility=_assessment_for(obj_id, LensType.TECHNICAL_FEASIBILITY),
                institutional_regional=_assessment_for(obj_id, LensType.INSTITUTIONAL_REGIONAL),
                financial_adoption=_assessment_for(obj_id, LensType.FINANCIAL_ADOPTION),
            )
        )
    return Stage4EvaluationBundle(
        case_id="case__threat__pmt",
        evaluations=evaluations,
    ).model_dump(mode="json")


class ContextualStage4Tests(unittest.TestCase):
    def test_contextual_system_prompt_rule_numbers_are_contiguous(self):
        expected = list(range(1, 27))
        for lens_name, prompt in CONTEXT_SYSTEM_PROMPT_BY_LENS.items():
            with self.subTest(lens=lens_name):
                actual = [
                    int(match.group(1))
                    for match in re.finditer(r"(?m)^\s*(\d+)\.\s", prompt)
                ]
                self.assertEqual(actual, expected)

    def test_jurisdiction_neutral_equivalence_key_ignores_region_label(self):
        kr = ContextualDecisionObject.model_validate(
            _contextual_object("KR", "Republic of Korea", "30")
        )
        eu = ContextualDecisionObject.model_validate(
            _contextual_object("EU", "European Union", "31")
        )
        for lens in LensType:
            self.assertEqual(
                context_lens_equivalence_key(kr, lens),
                context_lens_equivalence_key(eu, lens),
            )
            self.assertEqual(context_lens_equivalence_key(kr, lens)[0], "JURISDICTION_NEUTRAL")

    def test_institutional_regulatory_evidence_makes_equivalence_jurisdiction_bound(self):
        objects = []
        for code, region, legal_source in (
            ("KR", "Republic of Korea", "30"),
            ("EU", "European Union", "31"),
        ):
            value = _contextual_object(code, region, legal_source)
            value["decision_evidence_pack"]["slot_coverage"] = {
                "E3": ["regulatory_applicability"]
            }
            value["decision_evidence_pack"]["retrieval_metadata"]["covered_slots"] = [
                "regulatory_applicability"
            ]
            value["decision_evidence_pack"]["retrieval_metadata"]["missing_slots"] = sorted(
                set(DECISION_EVIDENCE_SLOT_CLAIM_TYPES) - {"regulatory_applicability"}
            )
            objects.append(ContextualDecisionObject.model_validate(value))
        kr_key = context_lens_equivalence_key(objects[0], LensType.INSTITUTIONAL_REGIONAL)
        eu_key = context_lens_equivalence_key(objects[1], LensType.INSTITUTIONAL_REGIONAL)
        self.assertEqual(kr_key[0], "JURISDICTION_BOUND")
        self.assertEqual(eu_key[0], "JURISDICTION_BOUND")
        self.assertNotEqual(kr_key, eu_key)

    def test_neutral_equivalence_prompt_masks_jurisdiction_names(self):
        objects = [
            ContextualDecisionObject.model_validate(_contextual_object("KR", "Republic of Korea", "30")),
            ContextualDecisionObject.model_validate(_contextual_object("EU", "European Union", "31")),
            ContextualDecisionObject.model_validate(_contextual_object("US", "United States", "32")),
        ]
        lens = LensType.TECHNICAL_FEASIBILITY
        key = context_lens_equivalence_key(objects[0], lens)
        synthetic_id = _equivalence_synthetic_id(
            decision_object=objects[0],
            lens_type=lens,
            equivalence_key=key,
        )
        payload, common_ids = _build_jurisdiction_neutral_payload(
            decision_object=objects[0],
            lens_type=lens,
            equivalence_members=objects,
            synthetic_id=synthetic_id,
        )
        serialized = str(payload)
        for forbidden in (
            "Republic of Korea",
            "European Union",
            "United States",
            "KR__CIS_IG2",
            "EU__CIS_IG2",
            "US__CIS_IG2",
        ):
            self.assertNotIn(forbidden, serialized)
        self.assertEqual(payload["decision_object"]["decision_object_id"], synthetic_id)
        self.assertTrue(common_ids)

    def test_neutral_equivalence_excludes_shared_jurisdiction_background_record(self):
        objects = []
        for code, region, legal_source in (
            ("KR", "Republic of Korea", "30"),
            ("EU", "European Union", "31"),
            ("US", "United States", "32"),
        ):
            value = _contextual_object(code, region, legal_source)
            regional_background = _decision_record()
            regional_background["evidence_id"] = "E_REGIONAL_BACKGROUND"
            regional_background["source"] = "European Union Primary Source: EUR-Lex"
            regional_background["content"] = "Background legal material not eligible for technical direction."
            value["forecast_evidence_ids"].append(regional_background["evidence_id"])
            value["evaluation_evidence_ids"].append(regional_background["evidence_id"])
            value["evaluation_evidence_records"].append(regional_background)
            objects.append(ContextualDecisionObject.model_validate(value))

        lens = LensType.TECHNICAL_FEASIBILITY
        key = context_lens_equivalence_key(objects[0], lens)
        synthetic_id = _equivalence_synthetic_id(
            decision_object=objects[0],
            lens_type=lens,
            equivalence_key=key,
        )
        payload, neutral_ids = _build_jurisdiction_neutral_payload(
            decision_object=objects[0],
            lens_type=lens,
            equivalence_members=objects,
            synthetic_id=synthetic_id,
        )
        self.assertNotIn("E_REGIONAL_BACKGROUND", neutral_ids)
        self.assertNotIn("European Union", str(payload))

    def test_neutral_equivalence_rejects_jurisdiction_bearing_directional_record(self):
        objects = []
        for code, region, legal_source in (
            ("KR", "Republic of Korea", "30"),
            ("EU", "European Union", "31"),
            ("US", "United States", "32"),
        ):
            value = _contextual_object(code, region, legal_source)
            for record in value["evaluation_evidence_records"]:
                if record.get("evidence_id") == "E3":
                    record["source"] = "European Union Primary Source: EUR-Lex"
            for record in value["decision_evidence_pack"]["decision_evidence_records"]:
                if record.get("evidence_id") == "E3":
                    record["source"] = "European Union Primary Source: EUR-Lex"
            objects.append(ContextualDecisionObject.model_validate(value))

        lens = LensType.TECHNICAL_FEASIBILITY
        key = context_lens_equivalence_key(objects[0], lens)
        synthetic_id = _equivalence_synthetic_id(
            decision_object=objects[0],
            lens_type=lens,
            equivalence_key=key,
        )
        with self.assertRaisesRegex(Exception, "jurisdiction-bearing directional evidence"):
            _build_jurisdiction_neutral_payload(
                decision_object=objects[0],
                lens_type=lens,
                equivalence_members=objects,
                synthetic_id=synthetic_id,
            )

    def test_stage4_identity_binds_jurisdiction_neutral_generation_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage3_artifact, _ = freeze_stage3_context_artifact(
                project_root=tmp,
                evidence_pack=_evidence_pack(),
                base_stage3_artifact=_stage3_artifact(),
                contextual_decision_bundle=_contextual_bundle(),
            )
            identity = build_stage4_context_identity(stage3_context_artifact=stage3_artifact)
        self.assertEqual(
            identity["generation_contract_version"],
            STAGE4_CONTEXT_GENERATION_CONTRACT_VERSION,
        )

    def test_bundle_rejects_jurisdiction_only_stance_drift_with_same_lens_evidence(self):
        objects = []
        for code, region, legal_source in (
            ("KR", "Republic of Korea", "30"),
            ("EU", "European Union", "31"),
            ("US", "United States", "32"),
        ):
            value = _contextual_object(code, region, legal_source)
            support = _decision_record()
            support["evidence_id"] = "E3"
            support["content"] = "Action-specific evidence supports technical feasibility."
            challenge = _decision_record()
            challenge["evidence_id"] = "E4"
            challenge["content"] = "Action-specific evidence challenges production feasibility."
            value["decision_evidence_ids"] = ["E3", "E4"]
            value["evaluation_evidence_ids"] = ["E1", "E2", "CTX1", "E3", "E4"]
            value["evaluation_evidence_records"] = [
                *_evidence_pack()["evidence"],
                _context_record(),
                support,
                challenge,
            ]
            value["decision_evidence_pack"]["decision_evidence_records"] = [support, challenge]
            value["decision_evidence_pack"]["decision_evidence_ids"] = ["E3", "E4"]
            value["decision_evidence_pack"]["slot_coverage"] = {
                "E3": ["technical_enablement"],
                "E4": ["technical_limitation"],
            }
            value["decision_evidence_pack"]["retrieval_metadata"]["covered_slots"] = [
                "technical_enablement",
                "technical_limitation",
            ]
            value["decision_evidence_pack"]["retrieval_metadata"]["missing_slots"] = sorted(
                set(DECISION_EVIDENCE_SLOT_CLAIM_TYPES)
                - {"technical_enablement", "technical_limitation"}
            )
            objects.append(ContextualDecisionObject.model_validate(value))

        evaluations = []
        for index, obj in enumerate(objects):
            technical = {
                "decision_object_id": obj.decision_object_id,
                "lens_type": LensType.TECHNICAL_FEASIBILITY.value,
                "stance": 0 if index == 0 else 1,
                "evidence_sufficiency": "PARTIAL",
                "rationale": "Both directions are grounded; the weighting differs only for the test.",
                "claims": [
                    {
                        "claim_id": f"T_SUPPORT_{index}",
                        "statement": "The action has grounded technical support.",
                        "evidence_ids": ["E3"],
                        "status": "SUPPORTS_FEASIBILITY",
                        "scope": "GENERAL",
                    },
                    {
                        "claim_id": f"T_CHALLENGE_{index}",
                        "statement": "The action has grounded technical limitations.",
                        "evidence_ids": ["E4"],
                        "status": "CHALLENGES_FEASIBILITY",
                        "scope": "GENERAL",
                    },
                ],
                "constraints": [],
                "evidence_gaps": [],
                "conditional_requirements": [],
            }
            institutional = _assessment_for(obj.decision_object_id, LensType.INSTITUTIONAL_REGIONAL)
            financial = _assessment_for(obj.decision_object_id, LensType.FINANCIAL_ADOPTION)
            evaluations.append(
                DecisionLensEvaluation(
                    decision_object_id=obj.decision_object_id,
                    technical_feasibility=technical,
                    institutional_regional=institutional,
                    financial_adoption=financial,
                )
            )

        bundle = Stage4EvaluationBundle(case_id="case__threat__pmt", evaluations=evaluations)
        with self.assertRaisesRegex(Exception, "Jurisdiction-label sensitivity"):
            validate_context_stage4_bundle(bundle, decision_objects=objects)

    def test_contextual_payload_contains_exact_union_and_hides_other_outputs(self):
        obj = ContextualDecisionObject.model_validate(_contextual_bundle()["contextual_decision_objects"][0])
        payload = build_context_stage4_evaluation_payload(
            decision_object=obj,
            expected_lens_type=LensType.INSTITUTIONAL_REGIONAL,
        )
        self.assertEqual(
            {item["evidence_id"] for item in payload["evaluation_evidence"]},
            {"E1", "E2", "CTX1", "E3"},
        )
        self.assertFalse(payload["evidence_boundary"]["sibling_lens_outputs_visible"])
        self.assertFalse(payload["evidence_boundary"]["other_scenario_outputs_visible"])
        institutional_contract = payload["directional_grounding_contract"]["institutional_regional"]
        self.assertIn(
            "governance_enablement",
            institutional_contract["GENERAL"]["eligible_slots"],
        )
        self.assertIn(
            "governance_enablement",
            institutional_contract["DEPLOYMENT_SPECIFIC"]["eligible_slots"],
        )
        self.assertEqual(
            institutional_contract["DEPLOYMENT_SPECIFIC"]["required_scope_slots"],
            ["regulatory_applicability"],
        )
        self.assertEqual(
            set(payload["directional_grounding_contract"]),
            {"institutional_regional"},
        )

    def test_contextual_prompt_projection_does_not_repeat_frozen_evidence_records(self):
        obj = ContextualDecisionObject.model_validate(_contextual_bundle()["contextual_decision_objects"][0])
        payload = build_context_stage4_evaluation_payload(
            decision_object=obj,
            expected_lens_type=LensType.TECHNICAL_FEASIBILITY,
        )
        prompt_object = payload["decision_object"]
        self.assertNotIn("evaluation_evidence_records", prompt_object)
        self.assertNotIn("decision_evidence_pack", prompt_object)
        self.assertNotIn("evaluation_evidence_ids", prompt_object)
        self.assertNotIn("decision_evidence_ids", prompt_object)
        self.assertNotIn("context_evidence_ids", prompt_object)
        self.assertNotIn("forecast_evidence_ids", prompt_object)
        for record in payload["evaluation_evidence"]:
            self.assertNotIn("content_hash", record)
            self.assertNotIn("source_content_sha256", record)
            self.assertNotIn("allowed_claim_types", record)
            self.assertNotIn("prohibited_claim_types", record)
            self.assertNotIn("evidence_chain_id", record)
            self.assertIn("evidence_id", record)
            self.assertIn("content", record)

    def test_contextual_schema_binds_object_lens_and_evidence_ids(self):
        obj = ContextualDecisionObject.model_validate(_contextual_bundle()["contextual_decision_objects"][0])
        schema = build_constrained_context_lens_response_schema(
            decision_object=obj,
            expected_lens_type=LensType.TECHNICAL_FEASIBILITY,
        )
        self.assertEqual(schema["properties"]["decision_object_id"]["const"], obj.decision_object_id)
        self.assertEqual(
            set(schema["$defs"]["LensClaim"]["properties"]["evidence_ids"]["items"]["enum"]),
            {"E1", "E2", "CTX1", "E3"},
        )
        self.assertEqual(set(schema["required"]), set(schema["properties"]))
        self.assertEqual(schema["properties"]["claims"]["minItems"], 1)

    def test_contextual_schema_requires_complete_top_level_arrays(self):
        obj = ContextualDecisionObject.model_validate(_contextual_bundle()["contextual_decision_objects"][0])
        schema = build_constrained_context_lens_response_schema(
            decision_object=obj,
            expected_lens_type=LensType.TECHNICAL_FEASIBILITY,
        )
        for field_name in (
            "claims",
            "constraints",
            "evidence_gaps",
            "conditional_requirements",
        ):
            self.assertIn(field_name, schema["required"])

    def test_contextual_bundle_requires_three_lens_evaluation_for_each_scenario(self):
        parsed = validate_context_stage4_bundle(
            _evaluation_bundle(),
            decision_objects=_contextual_bundle()["contextual_decision_objects"],
        )
        self.assertEqual(len(parsed.evaluations), 3)

    def test_outside_evidence_is_rejected(self):
        obj = ContextualDecisionObject.model_validate(_contextual_bundle()["contextual_decision_objects"][0])
        assessment = _assessment_for(obj.decision_object_id, LensType.TECHNICAL_FEASIBILITY)
        assessment["claims"][0]["evidence_ids"] = ["OUTSIDE"]
        with self.assertRaisesRegex(Exception, "outside contextual evaluation universe"):
            validate_context_lens_assessment(
                assessment,
                decision_object=obj,
                expected_lens_type=LensType.TECHNICAL_FEASIBILITY,
            )

    def test_deployment_specific_directional_claim_using_context_only_is_downgraded(self):
        obj = ContextualDecisionObject.model_validate(_contextual_bundle()["contextual_decision_objects"][0])
        assessment = _assessment_for(obj.decision_object_id, LensType.INSTITUTIONAL_REGIONAL)
        assessment.update(
            {
                "stance": 1,
                "evidence_sufficiency": "PARTIAL",
                "claims": [
                    {
                        "claim_id": "I_CTX_ONLY",
                        "statement": "The action is institutionally feasible in this deployment scenario.",
                        "evidence_ids": ["CTX1"],
                        "status": "SUPPORTS_FEASIBILITY",
                        "scope": "DEPLOYMENT_SPECIFIC",
                    }
                ],
                "evidence_gaps": [],
            }
        )
        with self.assertRaisesRegex(Exception, "cannot leave every feasibility claim NEUTRAL_CONTEXT"):
            validate_context_lens_assessment(
                assessment,
                decision_object=obj,
                expected_lens_type=LensType.INSTITUTIONAL_REGIONAL,
            )

    def test_directional_claim_from_noneligible_slot_is_conservatively_neutralized(self):
        obj = ContextualDecisionObject.model_validate(_contextual_bundle()["contextual_decision_objects"][0])
        assessment = _assessment_for(obj.decision_object_id, LensType.TECHNICAL_FEASIBILITY)
        assessment["claims"][0].update(
            {
                "evidence_ids": ["CTX1"],
                "status": "SUPPORTS_FEASIBILITY",
                "scope": "GENERAL",
            }
        )
        with self.assertRaisesRegex(Exception, "cannot leave every feasibility claim NEUTRAL_CONTEXT"):
            validate_context_lens_assessment(
                assessment,
                decision_object=obj,
                expected_lens_type=LensType.TECHNICAL_FEASIBILITY,
            )

    def test_general_directional_claim_using_forecast_only_is_downgraded(self):
        obj = ContextualDecisionObject.model_validate(_contextual_bundle()["contextual_decision_objects"][0])
        assessment = _assessment_for(obj.decision_object_id, LensType.TECHNICAL_FEASIBILITY)
        assessment["claims"][0].update(
            {
                "evidence_ids": ["E1"],
                "scope": "GENERAL",
            }
        )
        with self.assertRaisesRegex(Exception, "cannot leave every feasibility claim NEUTRAL_CONTEXT"):
            validate_context_lens_assessment(
                assessment,
                decision_object=obj,
                expected_lens_type=LensType.TECHNICAL_FEASIBILITY,
            )

    def test_general_supported_judgment_with_residual_context_gap_normalizes_to_partial(self):
        obj = ContextualDecisionObject.model_validate(_contextual_bundle()["contextual_decision_objects"][0])
        assessment = _assessment_for(obj.decision_object_id, LensType.TECHNICAL_FEASIBILITY)
        assessment.update(
            {
                "stance": 1,
                "evidence_sufficiency": "INSUFFICIENT_CONTEXT",
                "evidence_gaps": ["Deployment-specific architecture is not supplied."],
            }
        )
        parsed = validate_context_lens_assessment(
            assessment,
            decision_object=obj,
            expected_lens_type=LensType.TECHNICAL_FEASIBILITY,
        )
        self.assertEqual(parsed.stance, 1)
        self.assertEqual(parsed.evidence_sufficiency.value, "PARTIAL")

    def test_institutional_general_governance_claim_can_support_direction(self):
        obj = _contextual_object_with_decision_slot("governance_enablement")
        assessment = _assessment_for(obj.decision_object_id, LensType.INSTITUTIONAL_REGIONAL)
        assessment.update(
            {
                "stance": 1,
                "evidence_sufficiency": "PARTIAL",
                "claims": [
                    {
                        "claim_id": "I_GOV_GENERAL",
                        "statement": "The action has a documented governance and audit pathway.",
                        "evidence_ids": ["E3"],
                        "status": "SUPPORTS_FEASIBILITY",
                        "scope": "GENERAL",
                    }
                ],
            }
        )
        parsed = validate_context_lens_assessment(
            assessment,
            decision_object=obj,
            expected_lens_type=LensType.INSTITUTIONAL_REGIONAL,
        )
        self.assertEqual(parsed.stance, 1)
        self.assertEqual(parsed.claims[0].status.value, "SUPPORTS_FEASIBILITY")

    def test_institutional_deployment_specific_governance_only_claim_is_neutralized(self):
        obj = _contextual_object_with_decision_slot("governance_enablement")
        assessment = _assessment_for(obj.decision_object_id, LensType.INSTITUTIONAL_REGIONAL)
        assessment.update(
            {
                "stance": -1,
                "evidence_sufficiency": "PARTIAL",
                "claims": [
                    {
                        "claim_id": "I_GOV_SPECIFIC",
                        "statement": "The action is legally constrained in the frozen jurisdiction.",
                        "evidence_ids": ["E3"],
                        "status": "CHALLENGES_FEASIBILITY",
                        "scope": "DEPLOYMENT_SPECIFIC",
                    }
                ],
            }
        )
        with self.assertRaisesRegex(Exception, "cannot leave every feasibility claim NEUTRAL_CONTEXT"):
            validate_context_lens_assessment(
                assessment,
                decision_object=obj,
                expected_lens_type=LensType.INSTITUTIONAL_REGIONAL,
            )

    def test_deployment_maturity_alone_cannot_carry_technical_direction(self):
        obj = _contextual_object_with_decision_slot("deployment_maturity")
        assessment = _assessment_for(obj.decision_object_id, LensType.TECHNICAL_FEASIBILITY)
        assessment.update(
            {
                "stance": 1,
                "evidence_sufficiency": "PARTIAL",
                "claims": [
                    {
                        "claim_id": "T_MATURITY_ONLY",
                        "statement": "Deployment maturity alone supports technical feasibility.",
                        "evidence_ids": ["E3"],
                        "status": "SUPPORTS_FEASIBILITY",
                        "scope": "GENERAL",
                    }
                ],
            }
        )
        with self.assertRaisesRegex(Exception, "cannot leave every feasibility claim NEUTRAL_CONTEXT"):
            validate_context_lens_assessment(
                assessment,
                decision_object=obj,
                expected_lens_type=LensType.TECHNICAL_FEASIBILITY,
            )

    def test_deployment_maturity_alone_cannot_carry_financial_direction(self):
        obj = _contextual_object_with_decision_slot("deployment_maturity")
        assessment = _assessment_for(obj.decision_object_id, LensType.FINANCIAL_ADOPTION)
        assessment.update(
            {
                "stance": -1,
                "evidence_sufficiency": "PARTIAL",
                "claims": [
                    {
                        "claim_id": "F_MATURITY_ONLY",
                        "statement": "Deployment maturity alone challenges adoption feasibility.",
                        "evidence_ids": ["E3"],
                        "status": "CHALLENGES_FEASIBILITY",
                        "scope": "GENERAL",
                    }
                ],
            }
        )
        with self.assertRaisesRegex(Exception, "cannot leave every feasibility claim NEUTRAL_CONTEXT"):
            validate_context_lens_assessment(
                assessment,
                decision_object=obj,
                expected_lens_type=LensType.FINANCIAL_ADOPTION,
            )

    def test_regulatory_applicability_alone_cannot_carry_institutional_direction(self):
        obj = _contextual_object_with_decision_slot("regulatory_applicability")
        assessment = _assessment_for(obj.decision_object_id, LensType.INSTITUTIONAL_REGIONAL)
        assessment.update(
            {
                "stance": -1,
                "evidence_sufficiency": "PARTIAL",
                "claims": [
                    {
                        "claim_id": "I_REG_ONLY",
                        "statement": "Applicability alone challenges institutional feasibility.",
                        "evidence_ids": ["E3"],
                        "status": "CHALLENGES_FEASIBILITY",
                        "scope": "DEPLOYMENT_SPECIFIC",
                    }
                ],
            }
        )
        with self.assertRaisesRegex(Exception, "cannot leave every feasibility claim NEUTRAL_CONTEXT"):
            validate_context_lens_assessment(
                assessment,
                decision_object=obj,
                expected_lens_type=LensType.INSTITUTIONAL_REGIONAL,
            )

    def test_neutral_partial_with_all_neutral_claims_requires_repair(self):
        obj = _contextual_object_with_decision_slots(
            {
                "E3": ["adoption_benefit"],
                "E4": ["adoption_burden"],
            }
        )
        assessment = _assessment_for(obj.decision_object_id, LensType.FINANCIAL_ADOPTION)
        assessment.update(
            {
                "stance": 0,
                "evidence_sufficiency": "PARTIAL",
                "claims": [
                    {
                        "claim_id": "F_NEUTRAL_BENEFIT",
                        "statement": "The supplied record documents an adoption benefit.",
                        "evidence_ids": ["E3"],
                        "status": "NEUTRAL_CONTEXT",
                        "scope": "GENERAL",
                    },
                    {
                        "claim_id": "F_NEUTRAL_BURDEN",
                        "statement": "The supplied record documents an adoption burden.",
                        "evidence_ids": ["E4"],
                        "status": "NEUTRAL_CONTEXT",
                        "scope": "GENERAL",
                    },
                ],
            }
        )
        with self.assertRaisesRegex(Exception, "cannot leave every feasibility claim NEUTRAL_CONTEXT"):
            validate_context_lens_assessment(
                assessment,
                decision_object=obj,
                expected_lens_type=LensType.FINANCIAL_ADOPTION,
            )

    def test_neutral_partial_with_both_grounded_directions_is_valid(self):
        obj = _contextual_object_with_decision_slots(
            {
                "E3": ["adoption_benefit"],
                "E4": ["adoption_burden"],
            }
        )
        assessment = _assessment_for(obj.decision_object_id, LensType.FINANCIAL_ADOPTION)
        assessment.update(
            {
                "stance": 0,
                "evidence_sufficiency": "PARTIAL",
                "claims": [
                    {
                        "claim_id": "F_SUPPORT",
                        "statement": "The supplied record supports adoption feasibility.",
                        "evidence_ids": ["E3"],
                        "status": "SUPPORTS_FEASIBILITY",
                        "scope": "GENERAL",
                    },
                    {
                        "claim_id": "F_CHALLENGE",
                        "statement": "The supplied record challenges adoption feasibility.",
                        "evidence_ids": ["E4"],
                        "status": "CHALLENGES_FEASIBILITY",
                        "scope": "GENERAL",
                    },
                ],
            }
        )
        parsed = validate_context_lens_assessment(
            assessment,
            decision_object=obj,
            expected_lens_type=LensType.FINANCIAL_ADOPTION,
        )
        self.assertEqual(parsed.stance, 0)
        self.assertEqual(parsed.evidence_sufficiency.value, "PARTIAL")

    def test_institutional_deployment_specific_direction_requires_scope_and_carrier(self):
        obj = _contextual_object_with_decision_slots(
            {
                "E3": ["compliance_constraint"],
                "E4": ["regulatory_applicability"],
            }
        )
        assessment = _assessment_for(obj.decision_object_id, LensType.INSTITUTIONAL_REGIONAL)
        assessment.update(
            {
                "stance": -1,
                "evidence_sufficiency": "PARTIAL",
                "claims": [
                    {
                        "claim_id": "I_REG_CONSTRAINED",
                        "statement": "An applicable requirement and a documented compliance constraint challenge feasibility.",
                        "evidence_ids": ["E3", "E4"],
                        "status": "CHALLENGES_FEASIBILITY",
                        "scope": "DEPLOYMENT_SPECIFIC",
                    }
                ],
            }
        )
        parsed = validate_context_lens_assessment(
            assessment,
            decision_object=obj,
            expected_lens_type=LensType.INSTITUTIONAL_REGIONAL,
        )
        self.assertEqual(parsed.stance, -1)
        self.assertEqual(parsed.claims[0].status.value, "CHALLENGES_FEASIBILITY")

    def test_neutral_stance_with_one_sided_grounded_claim_requires_explicit_repair(self):
        obj = _contextual_object_with_decision_slot("adoption_burden")
        assessment = _assessment_for(obj.decision_object_id, LensType.FINANCIAL_ADOPTION)
        assessment.update(
            {
                "stance": 0,
                "evidence_sufficiency": "PARTIAL",
                "claims": [
                    {
                        "claim_id": "F_BURDEN",
                        "statement": "The supplied implementation requirement creates adoption burden.",
                        "evidence_ids": ["E3"],
                        "status": "CHALLENGES_FEASIBILITY",
                        "scope": "GENERAL",
                    }
                ],
            }
        )
        with self.assertRaisesRegex(Exception, "stance=0 is inconsistent with one-sided grounded"):
            validate_context_lens_assessment(
                assessment,
                decision_object=obj,
                expected_lens_type=LensType.FINANCIAL_ADOPTION,
            )

    def test_single_repair_message_reports_both_context_and_stance_inconsistencies(self):
        obj = _contextual_object_with_decision_slot("adoption_burden")
        assessment = _assessment_for(obj.decision_object_id, LensType.FINANCIAL_ADOPTION)
        assessment.update(
            {
                "stance": 0,
                "evidence_sufficiency": "INSUFFICIENT_CONTEXT",
                "claims": [
                    {
                        "claim_id": "F_BURDEN_BOTH",
                        "statement": "The supplied implementation requirement creates adoption burden.",
                        "evidence_ids": ["E3"],
                        "status": "CHALLENGES_FEASIBILITY",
                        "scope": "GENERAL",
                    }
                ],
                "evidence_gaps": ["Exact organization-specific budget fit is not evidenced."],
            }
        )
        with self.assertRaises(Exception) as caught:
            validate_context_lens_assessment(
                assessment,
                decision_object=obj,
                expected_lens_type=LensType.FINANCIAL_ADOPTION,
            )
        message = str(caught.exception)
        self.assertIn("must be represented as PARTIAL", message)
        self.assertIn("stance=0 is inconsistent with one-sided grounded", message)

    def test_mixed_valid_and_invalid_directional_claims_keep_only_grounded_direction(self):
        obj = ContextualDecisionObject.model_validate(_contextual_bundle()["contextual_decision_objects"][0])
        assessment = _assessment_for(obj.decision_object_id, LensType.TECHNICAL_FEASIBILITY)
        assessment.update(
            {
                "stance": 1,
                "evidence_sufficiency": "INSUFFICIENT_CONTEXT",
                "evidence_gaps": ["Deployment-specific architecture is not supplied."],
                "claims": [
                    {
                        "claim_id": "VALID",
                        "statement": "Implementation guidance supports general feasibility.",
                        "evidence_ids": ["E3"],
                        "status": "SUPPORTS_FEASIBILITY",
                        "scope": "GENERAL",
                    },
                    {
                        "claim_id": "OVERREACH",
                        "statement": "Forecast evidence alone supports technical feasibility.",
                        "evidence_ids": ["E1"],
                        "status": "SUPPORTS_FEASIBILITY",
                        "scope": "GENERAL",
                    },
                ],
            }
        )
        parsed = validate_context_lens_assessment(
            assessment,
            decision_object=obj,
            expected_lens_type=LensType.TECHNICAL_FEASIBILITY,
        )
        self.assertEqual(parsed.stance, 1)
        self.assertEqual(parsed.evidence_sufficiency.value, "PARTIAL")
        statuses = {claim.claim_id: claim.status.value for claim in parsed.claims}
        self.assertEqual(statuses["VALID"], "SUPPORTS_FEASIBILITY")
        self.assertEqual(statuses["OVERREACH"], "NEUTRAL_CONTEXT")

    def test_deployment_specific_directional_claim_accepts_matching_decision_evidence_slot(self):
        obj = ContextualDecisionObject.model_validate(_contextual_bundle()["contextual_decision_objects"][0])
        assessment = _assessment_for(obj.decision_object_id, LensType.TECHNICAL_FEASIBILITY)
        assessment["claims"][0].update(
            {
                "evidence_ids": ["E3"],
                "scope": "DEPLOYMENT_SPECIFIC",
            }
        )
        parsed = validate_context_lens_assessment(
            assessment,
            decision_object=obj,
            expected_lens_type=LensType.TECHNICAL_FEASIBILITY,
        )
        self.assertEqual(parsed.claims[0].evidence_ids, ["E3"])

    def test_decision_evidence_pack_rejects_post_cutoff_record(self):
        value = _contextual_object("KR", "Republic of Korea", "30")
        value["decision_evidence_pack"]["decision_evidence_records"][0]["available_at"] = "2025-01-01"
        with self.assertRaisesRegex(Exception, "temporal violation"):
            ContextualDecisionObject.model_validate(value)

    def test_decision_evidence_pack_rejects_wrong_snapshot_record(self):
        value = _contextual_object("KR", "Republic of Korea", "30")
        value["decision_evidence_pack"]["decision_evidence_records"][0]["source_snapshot_id"] = "other-snapshot"
        with self.assertRaisesRegex(Exception, "source_snapshot_id mismatch"):
            ContextualDecisionObject.model_validate(value)

    def test_contextual_stage3_and_stage4_artifacts_freeze_and_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage3_artifact, stage3_path = freeze_stage3_context_artifact(
                project_root=tmp,
                evidence_pack=_evidence_pack(),
                base_stage3_artifact=_stage3_artifact(),
                contextual_decision_bundle=_contextual_bundle(),
            )
            loaded_stage3, _, loaded_stage3_path = load_frozen_stage3_context_artifact(
                project_root=tmp,
                evidence_pack=_evidence_pack(),
                base_stage3_artifact=_stage3_artifact(),
            )
            self.assertEqual(stage3_path, loaded_stage3_path)
            self.assertEqual(stage3_artifact["artifact_sha256"], loaded_stage3["artifact_sha256"])

            stage4_artifact, stage4_path = freeze_stage4_context_artifact(
                project_root=tmp,
                stage3_context_artifact=stage3_artifact,
                evaluation_bundle=_evaluation_bundle(),
            )
            loaded_stage4, _, loaded_stage4_path = load_frozen_stage4_context_artifact(
                project_root=tmp,
                stage3_context_artifact=stage3_artifact,
            )
            self.assertEqual(stage4_path, loaded_stage4_path)
            self.assertEqual(stage4_artifact["artifact_sha256"], loaded_stage4["artifact_sha256"])

    def test_contextual_stage4_identity_uses_dedicated_128k_total_two_slot_runtime_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage3_artifact, _ = freeze_stage3_context_artifact(
                project_root=tmp,
                evidence_pack=_evidence_pack(),
                base_stage3_artifact=_stage3_artifact(),
                contextual_decision_bundle=_contextual_bundle(),
            )
            identity = build_stage4_context_identity(
                stage3_context_artifact=stage3_artifact,
            )
        runtime = identity["runtime"]
        self.assertEqual(runtime["expected_context_size"], STAGE4_CONTEXT_SIZE)
        self.assertEqual(runtime["expected_context_size"], 65536)
        self.assertEqual(runtime["expected_parallel_slots"], STAGE4_PARALLEL_SLOTS)
        self.assertEqual(runtime["expected_parallel_slots"], 2)
        self.assertEqual(runtime["server_total_context_size"], STAGE4_SERVER_CONTEXT_SIZE)
        self.assertEqual(runtime["server_total_context_size"], 131072)
        self.assertEqual(runtime["client_request_concurrency"], STAGE4_CLIENT_CONCURRENCY)
        self.assertEqual(runtime["client_request_concurrency"], 2)
        self.assertEqual(
            runtime["stage4_context_runtime_profile_version"],
            STAGE4_CONTEXT_RUNTIME_PROFILE_VERSION,
        )
        self.assertEqual(runtime, get_stage4_context_runtime_profile())


if __name__ == "__main__":
    unittest.main()

