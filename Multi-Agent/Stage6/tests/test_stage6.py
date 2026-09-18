from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from itertools import product
from pathlib import Path

from Pipeline.graph import create_graph
from Stage2.tests.test_stage2 import _result, _round
from Stage4.schema import LensType
from Stage4.tests.test_context_stage4 import _contextual_bundle
from Stage5.builder import build_stage5_diagnostic_bundle
from Stage5.tests.test_stage5 import _bundle_with_stances, _evidence_pack
from Stage6.artifact import (
    build_stage6_identity,
    freeze_stage6_artifact,
    freeze_stage6_checkpoint,
    load_frozen_stage6_artifact,
    load_stage6_checkpoint,
    stage6_artifact_path,
    stage6_checkpoint_path,
)
from Stage6.builder import build_stage6_bundle, build_stage6_frames
from Stage6.prompts import STAGE6_SYNTHESIS_SYSTEM_PROMPT
from Stage6.runtime import STAGE6_CLIENT_CONCURRENCY, STAGE6_PARALLEL_SLOTS
from Stage6.schema import (
    DecisionSynthesis,
    Stage6ValidationError,
    SynthesisNarrative,
    _inline_report_evidence_ids,
    build_constrained_synthesis_narrative_schema,
    validate_synthesis_narrative,
)


def _hash(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _fixture(stances: tuple[int, int, int] = (-1, 1, 0)):
    stage3 = _contextual_bundle()
    stage4 = _bundle_with_stances(stances)
    stage5 = build_stage5_diagnostic_bundle(
        evidence_pack=_evidence_pack(),
        stage2_result=_result([_round(1)]),
        stage3_context_bundle=stage3,
        stage4_context_bundle=stage4,
    ).model_dump(mode="json")
    frames = build_stage6_frames(
        stage3_context_bundle=stage3,
        stage4_context_bundle=stage4,
        stage5_diagnostic_bundle=stage5,
    )
    return stage3, stage4, stage5, frames


def _valid_narrative(frame: dict) -> dict:
    frozen = frame["deterministic_output"]
    return {
        "decision_object_id": frozen["decision_object_id"],
        "decision_recommendation": frozen["decision_recommendation"],
        "strategic_intelligence_report": (
            "The frozen recommendation follows from the evidence-grounded feasibility and adoption assessments. "
            "The technical assessment traces its directional reasoning to [EVIDENCE:E3], the institutional assessment also "
            "uses [EVIDENCE:E3] for its evidence-backed judgment, and the financial/adoption assessment preserves the same "
            "auditable evidence trace through [EVIDENCE:E3]. The report retains the material support, challenge, and unresolved "
            "limitations rather than converting the three functional dimensions into a vote. It therefore explains "
            "the policy-conditional strategic implication while leaving deployment-specific gaps and conditions visible."
        ),
    }


def _upstream_artifacts(stage3: dict, stage4: dict, stage5: dict):
    def wrap(case_id: str, output_key: str, output: dict, bundle_key: str) -> dict:
        value = {
            "case_id": case_id,
            output_key: _hash(output),
            bundle_key: output,
        }
        value["artifact_sha256"] = _hash(value)
        return value

    case_id = stage3["case_id"]
    return (
        wrap(case_id, "stage3_contextual_output_sha256", stage3, "contextual_decision_bundle"),
        wrap(case_id, "stage4_output_sha256", stage4, "evaluation_bundle"),
        wrap(case_id, "stage5_output_sha256", stage5, "diagnostic_bundle"),
    )


class Stage6Tests(unittest.TestCase):
    def test_deterministic_frame_preserves_lens_partition_and_adds_policy_recommendation(self):
        _, _, _, frames = _fixture((-1, 1, 0))
        for frame in frames:
            frozen = frame["deterministic_output"]
            self.assertEqual(frozen["supporting_lenses"], ["institutional_regional"])
            self.assertEqual(frozen["dissenting_lenses"], ["technical_feasibility"])
            self.assertEqual(frozen["indeterminate_lenses"], ["financial_adoption"])
            self.assertEqual(frozen["decision_recommendation"], "DO_NOT_RECOMMEND")
            self.assertEqual(frozen["decision_rule_id"], "CRITICAL_GATE_CHALLENGED")
            self.assertEqual(frozen["decision_policy_version"], "stage6-decision-policy-v1")
            self.assertEqual(frozen["recommendation_scope"], "STRATEGIC_CANDIDATE_ACTION")
            self.assertEqual(
                frozen["recommendation_interpretation"],
                "POLICY_CONDITIONAL_DECISION_SUPPORT",
            )
            self.assertEqual(
                frozen["lens_set_role"],
                "PREDECLARED_FUNCTIONAL_DIMENSIONS_NOT_STATISTICAL_SAMPLE",
            )
            self.assertEqual(
                frozen["d_lens_interpretation"],
                "DESCRIPTIVE_WITHIN_SET_DISPERSION_NOT_POPULATION_UNCERTAINTY",
            )
            self.assertAlmostEqual(frozen["d_lens"], 0.666667, places=6)
            self.assertNotIn("final_stance", frozen)
        self.assertNotIn("final_stance", DecisionSynthesis.model_fields)

    def test_decision_policy_is_role_based_not_majority_vote(self):
        for stances in product((-1, 0, 1), repeat=3):
            technical, institutional, financial = stances
            if technical == -1 or institutional == -1:
                expected, rule = "DO_NOT_RECOMMEND", "CRITICAL_GATE_CHALLENGED"
            elif technical == 0 or institutional == 0:
                expected, rule = "HOLD", "CRITICAL_GATE_UNRESOLVED"
            elif financial == 1:
                expected, rule = "RECOMMEND", "ALL_GATES_SUPPORTED"
            elif financial == 0:
                expected, rule = "RECOMMEND_PILOT", "ADOPTION_GATE_UNRESOLVED"
            else:
                expected, rule = "HOLD", "ADOPTION_GATE_CHALLENGED"
            with self.subTest(stances=stances):
                _, _, _, frames = _fixture(stances)
                for frame in frames:
                    frozen = frame["deterministic_output"]
                    self.assertEqual(frozen["decision_recommendation"], expected)
                    self.assertEqual(frozen["decision_rule_id"], rule)

    def test_decision_basis_contract_separates_policy_trigger_from_qualifiers(self):
        cases = {
            (-1, 1, 0): (
                "CRITICAL_GATE_CHALLENGED",
                {("technical_feasibility", -1, "CHALLENGED")},
                set(),
            ),
            (1, 0, 1): (
                "CRITICAL_GATE_UNRESOLVED",
                {("institutional_regional", 0, "UNRESOLVED")},
                set(),
            ),
            (1, 1, 1): (
                "ALL_GATES_SUPPORTED",
                {("financial_adoption", 1, "SUPPORTED")},
                {
                    ("technical_feasibility", 1, "SUPPORTED"),
                    ("institutional_regional", 1, "SUPPORTED"),
                },
            ),
            (1, 1, 0): (
                "ADOPTION_GATE_UNRESOLVED",
                {("financial_adoption", 0, "UNRESOLVED")},
                {
                    ("technical_feasibility", 1, "SUPPORTED"),
                    ("institutional_regional", 1, "SUPPORTED"),
                },
            ),
            (1, 1, -1): (
                "ADOPTION_GATE_CHALLENGED",
                {("financial_adoption", -1, "CHALLENGED")},
                {
                    ("technical_feasibility", 1, "SUPPORTED"),
                    ("institutional_regional", 1, "SUPPORTED"),
                },
            ),
        }
        for stances, (rule, trigger_expected, prereq_expected) in cases.items():
            with self.subTest(stances=stances):
                _, _, _, frames = _fixture(stances)
                for frame in frames:
                    contract = frame["narrative_source"]["decision_basis_contract"]
                    self.assertEqual(contract["decision_rule_id"], rule)
                    trigger = {
                        (item["lens"], item["stance"], item["state"])
                        for item in contract["policy_trigger_lenses"]
                    }
                    prereq = {
                        (item["lens"], item["stance"], item["state"])
                        for item in contract["policy_prerequisite_lenses"]
                    }
                    self.assertEqual(trigger, trigger_expected)
                    self.assertEqual(prereq, prereq_expected)
                    self.assertEqual(
                        contract["trigger_logic"],
                        (
                            "ANY_LISTED_TRIGGER_IS_POLICY_SUFFICIENT"
                            if rule in {"CRITICAL_GATE_CHALLENGED", "CRITICAL_GATE_UNRESOLVED"}
                            else "TRIGGER_AND_ALL_PREREQUISITES_REQUIRED"
                        ),
                    )
                    self.assertEqual(
                        contract["evidence_gaps_role"],
                        "SCOPE_QUALIFIER_NOT_POLICY_TRIGGER",
                    )
                    self.assertEqual(
                        contract["upstream_conditions_role"],
                        "REASSESSMENT_CONTEXT_NOT_POLICY_TRIGGER",
                    )

    def test_prompt_distinguishes_policy_causality_from_evidence_gaps(self):
        lowered = STAGE6_SYNTHESIS_SYSTEM_PROMPT.casefold()
        self.assertIn("decision_basis_contract", lowered)
        self.assertIn("only the listed `policy_trigger_lenses`", lowered)
        self.assertIn("-1` means challenged", lowered)
        self.assertIn("0` means unresolved/indeterminate", lowered)
        self.assertIn("not that the lens \"challenges the recommendation\"", lowered)
        self.assertIn("not directly confirmed or challenged by the supplied evidence", lowered)
        self.assertIn("final_mediator_adjudications", lowered)
        self.assertIn("mixed lens as having neither support nor challenge", lowered)
        self.assertIn("5,000-10,000 characters", lowered)
        self.assertIn("never exceed 12,000 characters", lowered)
        self.assertIn("combine materially similar limitations", lowered)
        self.assertIn("one listed critical-gate state is sufficient by itself", lowered)

    def test_report_cannot_negate_single_critical_gate_policy_sufficiency(self):
        _, _, _, frames = _fixture((-1, 1, 0))
        frame = frames[0]
        value = _valid_narrative(frame)
        value["strategic_intelligence_report"] += (
            " The recommendation does not imply that one dimension alone is decisive."
        )
        with self.assertRaises(Stage6ValidationError):
            validate_synthesis_narrative(value, frame=frame)

        safe = _valid_narrative(frame)
        safe["strategic_intelligence_report"] += (
            " The technical gate is decisive under the frozen policy without implying universal primacy across all decision frameworks."
        )
        validate_synthesis_narrative(safe, frame=frame)

    def test_synthesis_frame_preserves_final_mediator_adjudications_for_forecast_reporting(self):
        _, _, _, frames = _fixture()
        for frame in frames:
            critique = frame["deterministic_output"]["critique_summary"]
            self.assertTrue(critique["final_mediator_adjudications"])
            self.assertIn("matched_resolved_claim_ids", critique)

    def test_partial_evidence_is_a_scope_qualifier_not_an_extra_vote(self):
        _, _, _, frames = _fixture((1, 1, 1))
        for frame in frames:
            frozen = frame["deterministic_output"]
            self.assertEqual(frozen["decision_recommendation"], "RECOMMEND")
            self.assertEqual(
                set(frozen["evidence_quality_summary"]["evidence_sufficiency_by_lens"].values()),
                {"PARTIAL"},
            )

    def test_prompt_requests_substantive_synthesis_without_answer_template(self):
        lowered = STAGE6_SYNTHESIS_SYSTEM_PROMPT.casefold()
        self.assertIn("strategic intelligence report", lowered)
        self.assertIn("what the evidence actually supports", lowered)
        self.assertIn("cite the supplied evidence id inline", lowered)
        self.assertNotIn("example:", lowered)
        self.assertNotIn("write exactly", lowered)

    def test_prompt_does_not_force_literal_lens_words_or_action_name(self):
        lowered = STAGE6_SYNTHESIS_SYSTEM_PROMPT.casefold()
        self.assertNotIn("using the words", lowered)
        self.assertNotIn("must name the frozen candidate action", lowered)
        self.assertIn("not as a statistical sample or a voting panel", lowered)
        self.assertIn("descriptive within-set dispersion", lowered)
        self.assertIn("conditional on the frozen decision policy", lowered)

        _, _, _, frames = _fixture()
        frame = frames[0]
        value = _valid_narrative(frame)
        parsed = validate_synthesis_narrative(value, frame=frame)
        self.assertEqual(
            parsed.strategic_intelligence_report,
            value["strategic_intelligence_report"],
        )

    def test_constrained_schema_binds_trace_fields(self):
        _, _, _, frames = _fixture()
        frame = frames[0]
        frozen = frame["deterministic_output"]
        schema = build_constrained_synthesis_narrative_schema(frame=frame)
        self.assertEqual(
            schema["properties"]["decision_object_id"]["const"],
            frozen["decision_object_id"],
        )
        self.assertEqual(
            schema["properties"]["decision_recommendation"]["const"],
            frozen["decision_recommendation"],
        )
        self.assertNotIn("cited_evidence_ids", schema["properties"])

    def test_narrative_fidelity_preserves_policy_decision_and_rejects_meta_or_new_numbers(self):
        _, _, _, frames = _fixture()
        frame = frames[0]
        valid = _valid_narrative(frame)
        parsed = validate_synthesis_narrative(valid, frame=frame)
        self.assertIsInstance(parsed, SynthesisNarrative)

        bad = dict(valid)
        bad["strategic_intelligence_report"] += " The majority determines the final stance."
        with self.assertRaisesRegex(Exception, "forbidden directive/meta"):
            validate_synthesis_narrative(bad, frame=frame)

        allowed = dict(valid)
        allowed["strategic_intelligence_report"] += (
            " This recommendation is not based on a majority vote; the frozen role-based policy applies."
        )
        validate_synthesis_narrative(allowed, frame=frame)

        bad = dict(valid)
        bad["decision_recommendation"] = "RECOMMEND"
        with self.assertRaisesRegex(Exception, "decision recommendation changed"):
            validate_synthesis_narrative(bad, frame=frame)

        bad = dict(valid)
        bad["strategic_intelligence_report"] += " This produces a 95% success rate."
        with self.assertRaisesRegex(Exception, "unsupported numeric"):
            validate_synthesis_narrative(bad, frame=frame)

        # Quantities copied from an exact inline-cited frozen evidence record
        # are allowed even when that source text is intentionally not injected
        # wholesale into the Stage 6 synthesis prompt.  Equivalent percentage
        # spellings normalize to the same provenance token.
        frame_with_numeric_source = json.loads(json.dumps(frame))
        frame_with_numeric_source["validation_context"]["evidence_numeric_tokens_by_id"]["E3"] = [
            "17%",
            "21%",
            "41%",
        ]
        grounded_numeric = dict(valid)
        grounded_numeric["strategic_intelligence_report"] += (
            " The cited source reports 21 percent, 17 percent, and 41% [EVIDENCE:E3]."
        )
        validate_synthesis_narrative(grounded_numeric, frame=frame_with_numeric_source)

        # A number present only in some other, uncited frozen record must not
        # become available to the report.
        frame_with_numeric_source["validation_context"]["evidence_numeric_tokens_by_id"]["E4"] = [
            "95%"
        ]
        uncited_numeric = dict(valid)
        uncited_numeric["strategic_intelligence_report"] += " This produces a 95% success rate."
        with self.assertRaisesRegex(Exception, "unsupported numeric"):
            validate_synthesis_narrative(uncited_numeric, frame=frame_with_numeric_source)

        bad = dict(valid)
        bad["strategic_intelligence_report"] = bad["strategic_intelligence_report"].replace(
            "[EVIDENCE:E3]", "the frozen evidence"
        )
        with self.assertRaisesRegex(Exception, "did not cite any frozen upstream evidence ID inline"):
            validate_synthesis_narrative(bad, frame=frame)

        bad = dict(valid)
        bad["strategic_intelligence_report"] += " Unsupported source [EVIDENCE:FAKE_ID]."
        with self.assertRaisesRegex(Exception, "outside the frozen upstream evidence universe"):
            validate_synthesis_narrative(bad, frame=frame)

    def test_grouped_repeated_evidence_markers_are_parsed_as_exact_citations(self):
        self.assertEqual(
            _inline_report_evidence_ids(
                "[EVIDENCE:DE_A, EVIDENCE:DE_B]"
            ),
            {"DE_A", "DE_B"},
        )
        _, _, _, frames = _fixture()
        frame = frames[0]
        value = _valid_narrative(frame)
        value["strategic_intelligence_report"] = value["strategic_intelligence_report"].replace(
            "[EVIDENCE:E3]",
            "[EVIDENCE:E3, EVIDENCE:E3]",
            1,
        )
        parsed = validate_synthesis_narrative(value, frame=frame)
        self.assertIsInstance(parsed, SynthesisNarrative)

    def test_malformed_self_correction_bracket_is_rejected(self):
        _, _, _, frames = _fixture()
        frame = frames[0]
        value = _valid_narrative(frame)
        value["strategic_intelligence_report"] += (
            " A malformed editorial note such as "
            "[EVIDENCE:NOT_A_REAL_ID is not cited; correct citation below] "
            "is not a citation. The valid trace remains [EVIDENCE:E3]."
        )
        with self.assertRaisesRegex(Exception, "malformed inline evidence citation"):
            validate_synthesis_narrative(value, frame=frame)

    def test_v11_checkpoint_is_revalidated_and_promoted_without_generation(self):
        stage3, stage4, stage5, frames = _fixture()
        upstream = _upstream_artifacts(stage3, stage4, stage5)
        frame = frames[0]
        narrative = _valid_narrative(frame)
        with tempfile.TemporaryDirectory() as tmp:
            legacy_identity = build_stage6_identity(
                stage3_context_artifact=upstream[0],
                stage4_context_artifact=upstream[1],
                stage5_artifact=upstream[2],
                semantic_validation_version="stage6-semantic-validation-v11",
            )
            legacy_path = stage6_checkpoint_path(
                project_root=tmp,
                identity=legacy_identity,
                decision_object_id=narrative["decision_object_id"],
            )
            output = SynthesisNarrative.model_validate(narrative).model_dump(mode="json")
            legacy_checkpoint = {
                "schema_version": "stage6-synthesis-checkpoint-v5",
                "case_id": legacy_identity["case_id"],
                "scenario_id": frame["deterministic_output"]["scenario_id"],
                "input_fingerprint": legacy_identity["input_fingerprint"],
                "decision_object_id": narrative["decision_object_id"],
                "created_at": "2026-09-16T00:00:00+00:00",
                "narrative_sha256": _hash(output),
                "narrative": output,
            }
            legacy_checkpoint["checkpoint_sha256"] = _hash(legacy_checkpoint)
            legacy_path.parent.mkdir(parents=True, exist_ok=True)
            legacy_path.write_text(
                json.dumps(legacy_checkpoint, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            loaded, current_path = load_stage6_checkpoint(
                project_root=tmp,
                stage3_context_artifact=upstream[0],
                stage4_context_artifact=upstream[1],
                stage5_artifact=upstream[2],
                decision_object_id=narrative["decision_object_id"],
            )
            current_identity = build_stage6_identity(
                stage3_context_artifact=upstream[0],
                stage4_context_artifact=upstream[1],
                stage5_artifact=upstream[2],
            )
            self.assertIsNotNone(loaded)
            self.assertTrue(current_path.exists())
            self.assertEqual(loaded["input_fingerprint"], current_identity["input_fingerprint"])

    def test_v11_artifact_is_revalidated_and_promoted_without_generation(self):
        stage3, stage4, stage5, frames = _fixture()
        upstream = _upstream_artifacts(stage3, stage4, stage5)
        narratives = [_valid_narrative(frame) for frame in frames]
        bundle = build_stage6_bundle(
            case_id=stage5["case_id"],
            frames=frames,
            narratives=narratives,
        ).model_dump(mode="json")
        with tempfile.TemporaryDirectory() as tmp:
            legacy_identity = build_stage6_identity(
                stage3_context_artifact=upstream[0],
                stage4_context_artifact=upstream[1],
                stage5_artifact=upstream[2],
                semantic_validation_version="stage6-semantic-validation-v11",
            )
            legacy_artifact = {
                **legacy_identity,
                "created_at": "2026-09-16T00:00:00+00:00",
                "stage6_output_sha256": _hash(bundle),
                "synthesis_bundle": bundle,
            }
            legacy_artifact["artifact_sha256"] = _hash(legacy_artifact)
            legacy_path = stage6_artifact_path(project_root=tmp, identity=legacy_identity)
            legacy_path.parent.mkdir(parents=True, exist_ok=True)
            legacy_path.write_text(
                json.dumps(legacy_artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            loaded, current_identity, current_path = load_frozen_stage6_artifact(
                project_root=tmp,
                stage3_context_artifact=upstream[0],
                stage4_context_artifact=upstream[1],
                stage5_artifact=upstream[2],
            )
            self.assertIsNotNone(loaded)
            self.assertTrue(current_path.exists())
            self.assertEqual(loaded["input_fingerprint"], current_identity["input_fingerprint"])

    def test_report_may_cite_frozen_upstream_evidence_outside_stage4_claim_subset(self):
        _, _, _, frames = _fixture()
        frame = frames[0]
        claim_ids = {
            evidence_id
            for assessment in frame["deterministic_output"]["lens_evidence_trace"].values()
            for claim in assessment["claims"]
            for evidence_id in claim["evidence_ids"]
        }
        upstream_ids = set(frame["deterministic_output"]["provenance"]["evaluation_evidence_ids"])
        non_claim_ids = sorted(upstream_ids - claim_ids)
        self.assertTrue(non_claim_ids)

        value = _valid_narrative(frame)
        value["strategic_intelligence_report"] += (
            f" Additional frozen forecast/context evidence is traceable as "
            f"[EVIDENCE:{non_claim_ids[0]}]."
        )
        parsed = validate_synthesis_narrative(value, frame=frame)
        self.assertIsInstance(parsed, SynthesisNarrative)

    def test_narrative_fidelity_rejects_generic_draft_debug_residue(self):
        _, _, _, frames = _fixture()
        frame = frames[0]
        bad = _valid_narrative(frame)
        bad["strategic_intelligence_report"] = (
            "TODO placeholder text: this field has a typo and needs the correct key. "
            "This deliberately long filler remains draft material and is not a substantive strategic report. "
            "It repeats enough text to satisfy only the structural length boundary while retaining clear debug residue. "
            "The actual evidence marker [EVIDENCE:E3] is present solely so citation validation does not hide the hygiene failure."
        )
        with self.assertRaisesRegex(Exception, "draft/debug/placeholder residue"):
            validate_synthesis_narrative(bad, frame=frame)

    def test_gaps_and_upstream_conditions_are_preserved_without_becoming_the_policy(self):
        _, _, _, frames = _fixture((1, 1, 0))
        for frame in frames:
            frozen = frame["deterministic_output"]
            self.assertGreater(len(frozen["unresolved_evidence_gaps"]), 0)
            self.assertIsInstance(frozen["upstream_conditions"], list)
            self.assertEqual(frozen["decision_recommendation"], "RECOMMEND_PILOT")
            self.assertEqual(
                frozen["synthesis_authority"],
                "DETERMINISTIC_POLICY_WITH_EVIDENCE_GROUNDED_LLM_REPORT",
            )

    def test_final_output_preserves_full_stage4_lens_evidence_trace(self):
        _, _, _, frames = _fixture((-1, 1, 0))
        for frame in frames:
            trace = frame["deterministic_output"]["lens_evidence_trace"]
            self.assertEqual(
                set(trace),
                {"technical_feasibility", "institutional_regional", "financial_adoption"},
            )
            self.assertEqual(trace["technical_feasibility"]["stance"], -1)
            self.assertEqual(trace["institutional_regional"]["stance"], 1)
            self.assertEqual(trace["financial_adoption"]["stance"], 0)
            self.assertTrue(trace["technical_feasibility"]["claims"])
            self.assertTrue(trace["institutional_regional"]["claims"])
            self.assertTrue(trace["financial_adoption"]["claims"])

    def test_stage6_bundle_contains_model_generated_report_and_deterministic_evidence_trace(self):
        _, _, _, frames = _fixture()
        narratives = [_valid_narrative(frame) for frame in frames]
        bundle = build_stage6_bundle(
            case_id="case__threat__pmt",
            frames=frames,
            narratives=narratives,
        )
        self.assertEqual(len(bundle.syntheses), 3)
        for synthesis, frame in zip(bundle.syntheses, frames, strict=True):
            frozen = frame["deterministic_output"]
            payload = synthesis.model_dump(mode="json")
            for key, value in frozen.items():
                self.assertEqual(payload[key], value)
            self.assertEqual(
                payload["strategic_intelligence_report"],
                _valid_narrative(frame)["strategic_intelligence_report"],
            )
            self.assertEqual(payload["report_cited_evidence_ids"], ["E3"])

    def test_checkpoint_and_artifact_round_trip(self):
        stage3, stage4, stage5, frames = _fixture()
        a3, a4, a5 = _upstream_artifacts(stage3, stage4, stage5)
        narratives = [_valid_narrative(frame) for frame in frames]
        bundle = build_stage6_bundle(
            case_id="case__threat__pmt",
            frames=frames,
            narratives=narratives,
        )
        with tempfile.TemporaryDirectory() as tmp:
            frame = frames[0]
            decision_object_id = frame["deterministic_output"]["decision_object_id"]
            checkpoint_path = freeze_stage6_checkpoint(
                project_root=tmp,
                stage3_context_artifact=a3,
                stage4_context_artifact=a4,
                stage5_artifact=a5,
                decision_object_id=decision_object_id,
                narrative=narratives[0],
            )
            self.assertTrue(Path(checkpoint_path).exists())
            checkpoint, _ = load_stage6_checkpoint(
                project_root=tmp,
                stage3_context_artifact=a3,
                stage4_context_artifact=a4,
                stage5_artifact=a5,
                decision_object_id=decision_object_id,
            )
            self.assertEqual(checkpoint["narrative"]["decision_object_id"], decision_object_id)

            artifact, artifact_path = freeze_stage6_artifact(
                project_root=tmp,
                stage3_context_artifact=a3,
                stage4_context_artifact=a4,
                stage5_artifact=a5,
                synthesis_bundle=bundle,
            )
            self.assertTrue(Path(artifact_path).exists())
            loaded, identity, loaded_path = load_frozen_stage6_artifact(
                project_root=tmp,
                stage3_context_artifact=a3,
                stage4_context_artifact=a4,
                stage5_artifact=a5,
            )
            self.assertEqual(artifact_path, loaded_path)
            self.assertEqual(artifact["input_fingerprint"], identity["input_fingerprint"])
            self.assertEqual(loaded["stage6_output_sha256"], artifact["stage6_output_sha256"])

    def test_active_graph_ends_after_stage6_and_runtime_is_two_way(self):
        graph = create_graph().get_graph()
        edges = {(edge.source, edge.target) for edge in graph.edges}
        self.assertIn(("stage5_diagnostics", "stage6_prepare"), edges)
        self.assertIn(("stage6_prepare", "stage6_synthesis"), edges)
        self.assertIn(("stage6_synthesis", "stage6_complete"), edges)
        self.assertIn(("stage6_complete", "__end__"), edges)
        self.assertEqual(STAGE6_CLIENT_CONCURRENCY, 2)
        self.assertEqual(STAGE6_PARALLEL_SLOTS, 2)

        from Stage6.runtime import get_stage6_runtime_profile

        self.assertEqual(get_stage6_runtime_profile()["reasoning_effort"], "xhigh")


if __name__ == "__main__":
    unittest.main()

