import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


STAGE1_PARENT = Path(__file__).resolve().parents[2]
if str(STAGE1_PARENT) not in sys.path:
    sys.path.insert(0, str(STAGE1_PARENT))

import Stage1.nodes as nodes
from langchain_core.messages import HumanMessage, SystemMessage
from Pipeline.graph import create_graph
from Stage1.runtime import LlamaCppChatClient, get_runtime_config
from Stage1.artifact import (
    Stage1ArtifactError,
    build_stage1_identity,
    freeze_stage1_critic_checkpoint,
    freeze_stage1_artifact,
    load_frozen_stage1_artifact,
    load_stage1_critic_checkpoint,
)
from Stage1.cases import MAIN_STAGE1_CASES, main_case_set_manifest, validate_main_case_set
from Stage1.schema import (
    STAGE1_SEMANTIC_VALIDATION_VERSION,
    CriticAssessment,
    CriticType,
    Stage1ValidationError,
    build_constrained_critic_response_schema,
    validate_critic_assessment,
)
from Stage1.prompts import ATTACK_FEASIBILITY_SYSTEM_PROMPT, DEFENSE_ROBUSTNESS_SYSTEM_PROMPT


def _evidence_pack() -> dict:
    return {
        "case_id": "DDoS__NLP",
        "evidence": [
            {"evidence_id": "E001"},
            {"evidence_id": "E002"},
        ],
    }


def _modality_evidence_pack() -> dict:
    return {
        "case_id": "Targeted__AccessControl",
        "threat_id": "THREAT_TARGETED_ATTACK",
        "pmt_id": "PMT_ACCESS_CONTROL",
        "forecast_summary": {
            "threat_state": {"state_modality": "NoI"},
            "pmt_state": {"state_modality": "NoP"},
        },
        "evidence": [
            {
                "evidence_id": "E_TREND",
                "threat_ids": ["THREAT_TARGETED_ATTACK"],
                "pmt_ids": [],
                "allowed_claim_types": ["directional_threat_trend"],
                "content": "The number of targeted-attack incidents increased from 10 in 2022 to 14 in 2023.",
            },
            {
                "evidence_id": "E_MAGNITUDE",
                "threat_ids": ["THREAT_TARGETED_ATTACK"],
                "pmt_ids": [],
                "allowed_claim_types": ["directional_threat_trend"],
                "content": "Median attack bandwidth grew from 1.4 Gbps to 2.2 Gbps.",
            },
            {
                "evidence_id": "E_PUB",
                "threat_ids": [],
                "pmt_ids": ["PMT_ACCESS_CONTROL"],
                "allowed_claim_types": ["publication_metadata"],
            },
            {
                "evidence_id": "E_PUB_TREND",
                "threat_ids": [],
                "pmt_ids": ["PMT_ACCESS_CONTROL"],
                "allowed_claim_types": ["directional_publication_trend"],
            },
            {
                "evidence_id": "E_CONTROL",
                "threat_ids": [],
                "pmt_ids": ["PMT_ACCESS_CONTROL"],
                "allowed_claim_types": ["security_control"],
            },
        ],
    }


def _assessment(critic_type: str) -> dict:
    decisive_component = (
        "THREAT_TRAJECTORY" if critic_type == "attack_feasibility" else "PMT_TRAJECTORY"
    )
    return {
        "case_id": "DDoS__NLP",
        "critic_type": critic_type,
        "stance": 0,
        "evidence_sufficiency": "PARTIAL",
        "claims": [
            {
                "claim_id": "C1",
                "statement": "One supplied record supports the forecast.",
                "evidence_ids": ["E001"],
                "forecast_component": decisive_component,
                "forecast_relation": "SUPPORTS_FORECAST",
            },
            {
                "claim_id": "C2",
                "statement": "Another supplied record challenges the forecast.",
                "evidence_ids": ["E002"],
                "forecast_component": decisive_component,
                "forecast_relation": "CHALLENGES_FORECAST",
            },
        ],
        "stance_basis": {
            "supporting_claim_ids": ["C1"],
            "challenging_claim_ids": ["C2"],
            "decision": "BALANCED",
            "rationale": "The strongest directional findings are balanced.",
        },
        "unresolved_questions": ["Deployment evidence remains incomplete."],
        "unsupported_specificity_detected": False,
    }


class Stage1SchemaTests(unittest.TestCase):
    def test_model_reported_confidence_is_not_in_contract(self):
        self.assertNotIn("confidence", CriticAssessment.model_json_schema()["properties"])

    def test_valid_assessment_is_bound_to_stage0_case_and_evidence(self):
        parsed = validate_critic_assessment(
            _assessment("attack_feasibility"),
            evidence_pack=_evidence_pack(),
            expected_critic_type=CriticType.ATTACK_FEASIBILITY,
        )
        self.assertEqual(parsed.case_id, "DDoS__NLP")
        self.assertEqual(parsed.critic_type, CriticType.ATTACK_FEASIBILITY)

    def test_unknown_evidence_id_is_rejected(self):
        assessment = _assessment("attack_feasibility")
        assessment["claims"][0]["evidence_ids"] = ["E999"]
        with self.assertRaises(Stage1ValidationError):
            validate_critic_assessment(
                assessment,
                evidence_pack=_evidence_pack(),
                expected_critic_type=CriticType.ATTACK_FEASIBILITY,
            )

    def test_wrong_source_prefix_is_rejected_even_on_exact_digest_match(self):
        evidence_pack = {
            "case_id": "DDoS__NLP",
            "evidence": [
                {"evidence_id": "E_31_1C66ACA55FB23E5B"},
                {"evidence_id": "E002"},
            ],
        }
        assessment = _assessment("attack_feasibility")
        assessment["claims"][0]["evidence_ids"] = ["E_29_1C66ACA55FB23E5B"]
        with self.assertRaises(Stage1ValidationError):
            validate_critic_assessment(
                assessment,
                evidence_pack=evidence_pack,
                expected_critic_type=CriticType.ATTACK_FEASIBILITY,
            )

    def test_structured_generation_schema_hard_binds_exact_evidence_ids(self):
        schema = build_constrained_critic_response_schema(
            evidence_pack=_modality_evidence_pack(),
            expected_critic_type=CriticType.DEFENSE_ROBUSTNESS,
        )
        self.assertEqual(schema["properties"]["case_id"]["const"], "Targeted__AccessControl")
        self.assertEqual(schema["properties"]["critic_type"]["const"], "defense_robustness")
        allowed = schema["$defs"]["ClaimAssessment"]["properties"]["evidence_ids"]["items"]["enum"]
        self.assertEqual(allowed, ["E_CONTROL", "E_MAGNITUDE", "E_PUB", "E_PUB_TREND", "E_TREND"])
        self.assertNotIn("E_29_1C66ACA55FB23E5B", allowed)

    def test_wrong_critic_type_is_rejected(self):
        with self.assertRaises(Stage1ValidationError):
            validate_critic_assessment(
                _assessment("defense_robustness"),
                evidence_pack=_evidence_pack(),
                expected_critic_type=CriticType.ATTACK_FEASIBILITY,
            )

    def test_claim_requires_grounding_evidence(self):
        assessment = _assessment("attack_feasibility")
        claim = assessment["claims"][0]
        claim["evidence_ids"] = []
        with self.assertRaises(Stage1ValidationError):
            validate_critic_assessment(
                assessment,
                evidence_pack=_evidence_pack(),
                expected_critic_type=CriticType.ATTACK_FEASIBILITY,
            )

    def test_claims_cannot_be_empty(self):
        assessment = _assessment("attack_feasibility")
        assessment["claims"] = []
        with self.assertRaises(Stage1ValidationError):
            validate_critic_assessment(
                assessment,
                evidence_pack=_evidence_pack(),
                expected_critic_type=CriticType.ATTACK_FEASIBILITY,
            )

    def test_grounded_statement_can_challenge_forecast(self):
        assessment = _assessment("attack_feasibility")
        claim = assessment["claims"][0]
        claim["statement"] = "The supplied advisory reports no known public exploitation."
        claim["evidence_ids"] = ["E001"]
        claim["forecast_relation"] = "CHALLENGES_FORECAST"
        assessment["stance"] = -1
        assessment["stance_basis"] = {
            "supporting_claim_ids": [],
            "challenging_claim_ids": ["C1"],
            "decision": "CHALLENGE_DOMINATES",
            "rationale": "The challenge is decisive.",
        }
        parsed = validate_critic_assessment(
            assessment,
            evidence_pack=_evidence_pack(),
            expected_critic_type=CriticType.ATTACK_FEASIBILITY,
        )
        self.assertEqual(parsed.claims[0].forecast_relation.value, "CHALLENGES_FORECAST")

    def test_neutral_requires_explicit_balanced_basis(self):
        assessment = _assessment("attack_feasibility")
        assessment["stance_basis"]["challenging_claim_ids"] = []
        with self.assertRaises(Stage1ValidationError):
            validate_critic_assessment(
                assessment,
                evidence_pack=_evidence_pack(),
                expected_critic_type=CriticType.ATTACK_FEASIBILITY,
            )

    def test_attack_stance_basis_cannot_use_pmt_or_operational_component(self):
        assessment = _assessment("attack_feasibility")
        assessment["claims"][0]["forecast_component"] = "PMT_TRAJECTORY"
        with self.assertRaises(Stage1ValidationError):
            validate_critic_assessment(
                assessment,
                evidence_pack=_evidence_pack(),
                expected_critic_type=CriticType.ATTACK_FEASIBILITY,
            )

    def test_defense_stance_basis_cannot_use_threat_or_operational_component(self):
        assessment = _assessment("defense_robustness")
        assessment["claims"][0]["forecast_component"] = "THREAT_TRAJECTORY"
        with self.assertRaises(Stage1ValidationError):
            validate_critic_assessment(
                assessment,
                evidence_pack=_evidence_pack(),
                expected_critic_type=CriticType.DEFENSE_ROBUSTNESS,
            )

    def test_operational_directional_claim_does_not_block_no_directional_evidence(self):
        assessment = _assessment("defense_robustness")
        assessment["claims"] = [
            {
                "claim_id": "O1",
                "statement": "The supplied record describes an operational limitation.",
                "evidence_ids": ["E001"],
                "forecast_component": "OPERATIONAL_INTERPRETATION",
                "forecast_relation": "CHALLENGES_FORECAST",
            }
        ]
        assessment["stance"] = 0
        assessment["stance_basis"] = {
            "supporting_claim_ids": [],
            "challenging_claim_ids": [],
            "decision": "NO_DIRECTIONAL_EVIDENCE",
            "rationale": "The operational finding does not directly bear on the PMT forecast modality.",
        }
        parsed = validate_critic_assessment(
            assessment,
            evidence_pack=_evidence_pack(),
            expected_critic_type=CriticType.DEFENSE_ROBUSTNESS,
        )
        self.assertEqual(parsed.stance_basis.decision.value, "NO_DIRECTIONAL_EVIDENCE")

    def test_pmt_nop_trajectory_rejects_control_existence_as_direct_directional_evidence(self):
        assessment = {
            "case_id": "Targeted__AccessControl",
            "critic_type": "defense_robustness",
            "stance": 1,
            "evidence_sufficiency": "PARTIAL",
            "claims": [
                {
                    "claim_id": "P1",
                    "statement": "A control catalog describes access-control mechanisms.",
                    "evidence_ids": ["E_CONTROL"],
                    "forecast_component": "PMT_TRAJECTORY",
                    "forecast_relation": "SUPPORTS_FORECAST",
                }
            ],
            "stance_basis": {
                "supporting_claim_ids": ["P1"],
                "challenging_claim_ids": [],
                "decision": "SUPPORT_DOMINATES",
                "rationale": "The control is documented.",
            },
            "unresolved_questions": [],
            "unsupported_specificity_detected": False,
        }
        with self.assertRaisesRegex(Stage1ValidationError, "PMT_TRAJECTORY"):
            validate_critic_assessment(
                assessment,
                evidence_pack=_modality_evidence_pack(),
                expected_critic_type=CriticType.DEFENSE_ROBUSTNESS,
            )

    def test_pmt_nop_trajectory_rejects_static_publication_metadata(self):
        assessment = {
            "case_id": "Targeted__AccessControl",
            "critic_type": "defense_robustness",
            "stance": 1,
            "evidence_sufficiency": "PARTIAL",
            "claims": [
                {
                    "claim_id": "P1",
                    "statement": "A bibliographic record confirms an access-control publication exists.",
                    "evidence_ids": ["E_PUB"],
                    "forecast_component": "PMT_TRAJECTORY",
                    "forecast_relation": "SUPPORTS_FORECAST",
                }
            ],
            "stance_basis": {
                "supporting_claim_ids": ["P1"],
                "challenging_claim_ids": [],
                "decision": "SUPPORT_DOMINATES",
                "rationale": "The cited publication exists.",
            },
            "unresolved_questions": [],
            "unsupported_specificity_detected": False,
        }
        with self.assertRaisesRegex(Stage1ValidationError, "directional_publication_trend"):
            validate_critic_assessment(
                assessment,
                evidence_pack=_modality_evidence_pack(),
                expected_critic_type=CriticType.DEFENSE_ROBUSTNESS,
            )

    def test_pmt_nop_trajectory_accepts_explicit_directional_publication_trend(self):
        assessment = {
            "case_id": "Targeted__AccessControl",
            "critic_type": "defense_robustness",
            "stance": 1,
            "evidence_sufficiency": "PARTIAL",
            "claims": [
                {
                    "claim_id": "P1",
                    "statement": "The supplied evidence directly reports increasing publication activity over time.",
                    "evidence_ids": ["E_PUB_TREND"],
                    "forecast_component": "PMT_TRAJECTORY",
                    "forecast_relation": "SUPPORTS_FORECAST",
                }
            ],
            "stance_basis": {
                "supporting_claim_ids": ["P1"],
                "challenging_claim_ids": [],
                "decision": "SUPPORT_DOMINATES",
                "rationale": "The evidence directly reports the NoP direction.",
            },
            "unresolved_questions": [],
            "unsupported_specificity_detected": False,
        }
        parsed = validate_critic_assessment(
            assessment,
            evidence_pack=_modality_evidence_pack(),
            expected_critic_type=CriticType.DEFENSE_ROBUSTNESS,
        )
        self.assertEqual(parsed.stance, 1)

    def test_threat_noi_trajectory_accepts_explicit_incident_count_trend(self):
        assessment = {
            "case_id": "Targeted__AccessControl",
            "critic_type": "attack_feasibility",
            "stance": 1,
            "evidence_sufficiency": "PARTIAL",
            "claims": [
                {
                    "claim_id": "T1",
                    "statement": "The number of targeted-attack incidents increased from 10 in 2022 to 14 in 2023.",
                    "evidence_ids": ["E_TREND"],
                    "forecast_component": "THREAT_TRAJECTORY",
                    "forecast_relation": "SUPPORTS_FORECAST",
                }
            ],
            "stance_basis": {
                "supporting_claim_ids": ["T1"],
                "challenging_claim_ids": [],
                "decision": "SUPPORT_DOMINATES",
                "rationale": "The evidence directly reports the incident-count direction.",
            },
            "unresolved_questions": [],
            "unsupported_specificity_detected": False,
        }
        parsed = validate_critic_assessment(
            assessment,
            evidence_pack=_modality_evidence_pack(),
            expected_critic_type=CriticType.ATTACK_FEASIBILITY,
        )
        self.assertEqual(parsed.stance, 1)

    def test_threat_noi_trajectory_rejects_attack_magnitude_as_incident_direction(self):
        assessment = {
            "case_id": "Targeted__AccessControl",
            "critic_type": "attack_feasibility",
            "stance": 1,
            "evidence_sufficiency": "PARTIAL",
            "claims": [
                {
                    "claim_id": "T1",
                    "statement": "Median attack bandwidth grew from 1.4 Gbps to 2.2 Gbps.",
                    "evidence_ids": ["E_MAGNITUDE"],
                    "forecast_component": "THREAT_TRAJECTORY",
                    "forecast_relation": "SUPPORTS_FORECAST",
                }
            ],
            "stance_basis": {
                "supporting_claim_ids": ["T1"],
                "challenging_claim_ids": [],
                "decision": "SUPPORT_DOMINATES",
                "rationale": "Attack magnitude increased.",
            },
            "unresolved_questions": [],
            "unsupported_specificity_detected": False,
        }
        with self.assertRaisesRegex(Stage1ValidationError, "incident count or frequency"):
            validate_critic_assessment(
                assessment,
                evidence_pack=_modality_evidence_pack(),
                expected_critic_type=CriticType.ATTACK_FEASIBILITY,
            )

    def test_threat_noi_trajectory_rejects_intrusion_share_as_incident_direction(self):
        evidence_pack = _modality_evidence_pack()
        evidence_pack["evidence"].append(
            {
                "evidence_id": "E_SHARE",
                "threat_ids": ["THREAT_TARGETED_ATTACK"],
                "pmt_ids": [],
                "allowed_claim_types": ["directional_threat_trend"],
                "content": "Targeted activity occurred in 3% of intrusions in 2020, 4% in 2021, and 6% in 2022.",
            }
        )
        assessment = {
            "case_id": "Targeted__AccessControl",
            "critic_type": "attack_feasibility",
            "stance": 1,
            "evidence_sufficiency": "PARTIAL",
            "claims": [
                {
                    "claim_id": "T1",
                    "statement": "Targeted activity rose from 3% to 6% of intrusions.",
                    "evidence_ids": ["E_SHARE"],
                    "forecast_component": "THREAT_TRAJECTORY",
                    "forecast_relation": "SUPPORTS_FORECAST",
                }
            ],
            "stance_basis": {
                "supporting_claim_ids": ["T1"],
                "challenging_claim_ids": [],
                "decision": "SUPPORT_DOMINATES",
                "rationale": "The intrusion share increased.",
            },
            "unresolved_questions": [],
            "unsupported_specificity_detected": False,
        }
        with self.assertRaisesRegex(Stage1ValidationError, "incident count or frequency"):
            validate_critic_assessment(
                assessment,
                evidence_pack=evidence_pack,
                expected_critic_type=CriticType.ATTACK_FEASIBILITY,
            )

    def test_gap_direction_requires_direct_evidence_for_both_forecast_state_sides(self):
        assessment = {
            "case_id": "Targeted__AccessControl",
            "critic_type": "defense_robustness",
            "stance": 1,
            "evidence_sufficiency": "PARTIAL",
            "claims": [
                {
                    "claim_id": "G1",
                    "statement": "Threat activity evidence alone is used to infer the gap.",
                    "evidence_ids": ["E_TREND"],
                    "forecast_component": "GAP_DIRECTION",
                    "forecast_relation": "SUPPORTS_FORECAST",
                }
            ],
            "stance_basis": {
                "supporting_claim_ids": ["G1"],
                "challenging_claim_ids": [],
                "decision": "SUPPORT_DOMINATES",
                "rationale": "The threat side is directional.",
            },
            "unresolved_questions": [],
            "unsupported_specificity_detected": False,
        }
        with self.assertRaisesRegex(Stage1ValidationError, "both forecast-state sides"):
            validate_critic_assessment(
                assessment,
                evidence_pack=_modality_evidence_pack(),
                expected_critic_type=CriticType.DEFENSE_ROBUSTNESS,
            )

    def test_combined_semantic_error_reports_modality_and_unknown_id_together(self):
        assessment = {
            "case_id": "Targeted__AccessControl",
            "critic_type": "defense_robustness",
            "stance": 1,
            "evidence_sufficiency": "PARTIAL",
            "claims": [
                {
                    "claim_id": "P1",
                    "statement": "A control catalog describes access-control mechanisms.",
                    "evidence_ids": ["E_CONTROL", "E_UNKNOWN"],
                    "forecast_component": "PMT_TRAJECTORY",
                    "forecast_relation": "SUPPORTS_FORECAST",
                }
            ],
            "stance_basis": {
                "supporting_claim_ids": ["P1"],
                "challenging_claim_ids": [],
                "decision": "SUPPORT_DOMINATES",
                "rationale": "The control is documented.",
            },
            "unresolved_questions": [],
            "unsupported_specificity_detected": False,
        }
        with self.assertRaises(Stage1ValidationError) as caught:
            validate_critic_assessment(
                assessment,
                evidence_pack=_modality_evidence_pack(),
                expected_critic_type=CriticType.DEFENSE_ROBUSTNESS,
            )
        message = str(caught.exception)
        self.assertIn("E_UNKNOWN", message)
        self.assertIn("PMT_TRAJECTORY", message)


class _FakeStructuredRunner:
    def __init__(self, result: dict, semantic_validator=None):
        self.result = CriticAssessment.model_validate(result)
        self.messages = None
        self.semantic_validator = semantic_validator

    async def ainvoke(self, messages):
        self.messages = messages
        return self.semantic_validator(self.result) if self.semantic_validator else self.result


class _FakeLLM:
    def __init__(self, result: dict):
        self.result = result
        self.runner = None

    def with_structured_output(
        self,
        schema,
        method="json_schema",
        progress_label=None,
        semantic_validator=None,
        response_schema=None,
    ):
        self.schema = schema
        self.method = method
        self.progress_label = progress_label
        self.response_schema = response_schema
        self.runner = _FakeStructuredRunner(self.result, semantic_validator=semantic_validator)
        return self.runner


class Stage1NodeTests(unittest.IsolatedAsyncioTestCase):
    async def test_attack_and_defense_initial_prompts_are_independent(self):
        state = {
            "forecast_data": "STAGE0_PAYLOAD_ONLY",
            "evidence_pack": _evidence_pack(),
            "attack_assessment": {"sentinel": "ATTACK_OUTPUT_MUST_NOT_BE_READ_BY_DEFENSE"},
            "defense_assessment": {"sentinel": "DEFENSE_OUTPUT_MUST_NOT_BE_READ_BY_ATTACK"},
            "iteration_count": 0,
            "messages": [],
        }

        attack_llm = _FakeLLM(_assessment("attack_feasibility"))
        with (
            patch.object(nodes, "get_llm", return_value=attack_llm),
            patch.object(nodes, "freeze_stage1_critic_checkpoint", return_value=Path("attack-checkpoint.json")),
        ):
            await nodes.attack_feasibility_critic_node(state)
        attack_prompt = attack_llm.runner.messages[0].content
        self.assertIn("STAGE0_PAYLOAD_ONLY", attack_prompt)
        self.assertNotIn("DEFENSE_OUTPUT_MUST_NOT_BE_READ_BY_ATTACK", attack_prompt)
        self.assertEqual(
            attack_llm.response_schema["$defs"]["ClaimAssessment"]["properties"]["evidence_ids"]["items"]["enum"],
            ["E001", "E002"],
        )

        defense_llm = _FakeLLM(_assessment("defense_robustness"))
        with (
            patch.object(nodes, "get_llm", return_value=defense_llm),
            patch.object(nodes, "freeze_stage1_critic_checkpoint", return_value=Path("defense-checkpoint.json")),
        ):
            await nodes.defense_robustness_critic_node(state)
        defense_prompt = defense_llm.runner.messages[0].content
        self.assertIn("STAGE0_PAYLOAD_ONLY", defense_prompt)
        self.assertNotIn("ATTACK_OUTPUT_MUST_NOT_BE_READ_BY_DEFENSE", defense_prompt)

    async def test_frozen_artifact_skips_both_llm_calls(self):
        state = {
            "forecast_data": "STAGE0_PAYLOAD_ONLY",
            "evidence_pack": _evidence_pack(),
            "attack_assessment": _assessment("attack_feasibility"),
            "defense_assessment": _assessment("defense_robustness"),
            "stage1_artifact_reused": True,
            "messages": [],
        }
        with patch.object(nodes, "get_llm", side_effect=AssertionError("LLM must not be called")):
            attack = await nodes.attack_feasibility_critic_node(state)
            defense = await nodes.defense_robustness_critic_node(state)
        self.assertIn("reused frozen", attack["messages"][0])
        self.assertIn("reused frozen", defense["messages"][0])

    async def test_exact_critic_checkpoint_skips_only_that_critic(self):
        state = {
            "forecast_data": "STAGE0_PAYLOAD_ONLY",
            "evidence_pack": _evidence_pack(),
            "attack_assessment": _assessment("attack_feasibility"),
            "defense_assessment": None,
            "stage1_artifact_reused": False,
            "attack_checkpoint_reused": True,
            "defense_checkpoint_reused": False,
            "messages": [],
        }
        with patch.object(nodes, "get_llm", side_effect=AssertionError("Attack LLM must not be called")):
            attack = await nodes.attack_feasibility_critic_node(state)
        self.assertIn("resumed from exact critic checkpoint", attack["messages"][0])

    async def test_stage1_join_requires_both_assessments(self):
        state = {
            "evidence_pack": _evidence_pack(),
            "attack_assessment": _assessment("attack_feasibility"),
            "defense_assessment": None,
        }
        with self.assertRaises(ValueError):
            nodes.stage1_complete_node(state)


class Stage1PromptTests(unittest.TestCase):
    def test_slot_progress_separates_input_generated_and_context_tokens(self):
        client = LlamaCppChatClient()

        class _Response:
            def raise_for_status(self):
                return None

            def json(self):
                return [
                    {
                        "state": True,
                        "id_task": 123,
                        "n_prompt_tokens": 21302,
                        "n_prompt_tokens_cache": 13454,
                        "n_prompt_tokens_processed": 4,
                        "next_token": [{"n_decoded": 7840, "n_remain": -1}],
                    }
                ]

        with patch("Stage1.runtime.requests.get", return_value=_Response()):
            progress = client._read_slot_progress()

        self.assertEqual(progress["input_tokens"], 13458)
        self.assertEqual(progress["generated_tokens"], 7840)
        self.assertEqual(progress["context_tokens"], 21302)
        self.assertEqual(progress["prompt_cache_tokens"], 13454)
        self.assertEqual(progress["prompt_processed_tokens"], 4)
        self.assertEqual(progress["remaining_tokens"], -1)
        self.assertEqual(progress["task_id"], 123)

    def test_slot_progress_does_not_treat_zero_cache_counter_as_zero_input(self):
        client = LlamaCppChatClient()

        class _Response:
            def raise_for_status(self):
                return None

            def json(self):
                return [
                    {
                        "state": True,
                        "id_task": 1,
                        "n_prompt_tokens": 10282,
                        "n_prompt_tokens_cache": 0,
                        "n_prompt_tokens_processed": 10154,
                        "next_token": [{"n_decoded": 128, "n_remain": -1}],
                    }
                ]

        with patch("Stage1.runtime.requests.get", return_value=_Response()):
            progress = client._read_slot_progress()

        self.assertEqual(progress["input_tokens"], 10154)
        self.assertEqual(progress["generated_tokens"], 128)
        self.assertEqual(progress["context_tokens"], 10282)
        self.assertEqual(progress["prompt_cache_tokens"], 0)
        self.assertEqual(progress["prompt_processed_tokens"], 10154)

    def test_prompts_remove_legacy_scenario_dependencies(self):
        self.assertNotIn("{defense_plan}", ATTACK_FEASIBILITY_SYSTEM_PROMPT)
        self.assertNotIn("{attack_plan}", DEFENSE_ROBUSTNESS_SYSTEM_PROMPT)
        self.assertIn("Do **not** design an attack scenario", ATTACK_FEASIBILITY_SYSTEM_PROMPT)
        self.assertIn("Do **not** create a counter-attack plan", DEFENSE_ROBUSTNESS_SYSTEM_PROMPT)
        for prompt in (ATTACK_FEASIBILITY_SYSTEM_PROMPT, DEFENSE_ROBUSTNESS_SYSTEM_PROMPT):
            self.assertIn("# Fixed Internal Review Order", prompt)
            ordered_markers = [
                "Evidence inventory and boundary",
                "Forecast decomposition",
                "Forecast-supporting evidence mapping",
                "Forecast-challenging evidence mapping",
                "Cross-source conflict and alternative-explanation audit",
                "Modality and causal-boundary audit",
                "Temporal consistency",
                "Claim polarity, specificity, and traceability audit",
                "Evidence-balance decision",
            ]
            positions = [prompt.index(marker) for marker in ordered_markers]
            self.assertEqual(positions, sorted(positions))
            self.assertIn("inspect every supplied evidence record at least once", prompt)
            self.assertNotIn("evidentiary_status", prompt)
            self.assertIn("forecast_component", prompt)
            self.assertIn("Preserve modality", prompt)
            self.assertIn("forecast_relation", prompt)
            self.assertIn("stance_basis", prompt)
            self.assertIn("claims` must never be empty", prompt)

    def test_qwen_thinking_sampling_matches_official_profile(self):
        config = get_runtime_config()
        self.assertEqual(config.model, "Qwen3.8-27B-Q6_K_L")
        self.assertEqual(config.temperature, 1.0)
        self.assertEqual(config.top_p, 0.95)
        self.assertEqual(config.top_k, 20)
        self.assertEqual(config.min_p, 0.0)
        self.assertEqual(config.presence_penalty, 0.0)
        self.assertEqual(config.repeat_penalty, 1.0)
        self.assertEqual(config.reasoning_effort, "xhigh")
        self.assertEqual(config.expected_model_sha256, "e8750bb81ba49f90eb68df99776b250dbc14666043098952de6422cbecd77a21")
        self.assertEqual(config.expected_llama_build, "b10919")
        self.assertEqual(config.expected_llama_commit, "d3146f2b5")
        self.assertEqual(config.expected_context_size, 65536)
        self.assertEqual(config.expected_parallel_slots, 2)
        self.assertEqual(config.expected_gpu_layers, "all")
        self.assertEqual(config.batch_size, 2048)
        self.assertEqual(config.ubatch_size, 512)
        self.assertEqual(config.mtp_draft_n_max, 4)
        self.assertEqual(config.mtp_draft_p_min, 0.05)
        self.assertTrue(config.mtp_enabled)
        self.assertEqual(config.structured_validation_retries, 1)

    def test_structured_output_semantic_failure_gets_one_fixed_repair(self):
        client = LlamaCppChatClient()
        invalid = _assessment("attack_feasibility")
        invalid["claims"][0]["evidence_ids"] = ["E999"]
        valid = _assessment("attack_feasibility")
        responses = [
            {"choices": [{"message": {"content": json.dumps(invalid)}}]},
            {"choices": [{"message": {"content": json.dumps(valid)}}]},
        ]
        with patch.object(client, "_post", side_effect=responses) as post, patch("builtins.print") as printed:
            parsed = client._invoke_structured(
                [SystemMessage(content="system"), HumanMessage(content="evaluate")],
                CriticAssessment,
                semantic_validator=lambda assessment: validate_critic_assessment(
                    assessment,
                    evidence_pack=_evidence_pack(),
                    expected_critic_type=CriticType.ATTACK_FEASIBILITY,
                ),
            )
        self.assertEqual(parsed.case_id, "DDoS__NLP")
        self.assertEqual(post.call_count, 2)
        repair_messages = post.call_args_list[1].args[0]["messages"]
        self.assertIn("failed deterministic schema/semantic validation", repair_messages[-1]["content"])
        self.assertIn("Never infer or repair an evidence ID from its digest", repair_messages[-1]["content"])
        logged = "\n".join(str(call) for call in printed.call_args_list)
        self.assertIn("structured schema/semantic validation failed", logged)
        self.assertIn("E999", logged)


class Stage1ArtifactTests(unittest.TestCase):
    def test_identity_binds_semantic_validation_version(self):
        identity = build_stage1_identity(
            evidence_pack=_evidence_pack(),
            forecast_data="FORECAST",
        )
        self.assertEqual(
            identity["semantic_validation_version"],
            STAGE1_SEMANTIC_VALIDATION_VERSION,
        )
        self.assertEqual(
            set(identity["constrained_output_schema_sha256"]),
            {"attack_feasibility", "defense_robustness"},
        )

    def test_exact_critic_checkpoint_can_resume_before_final_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = freeze_stage1_critic_checkpoint(
                project_root=tmp,
                evidence_pack=_evidence_pack(),
                forecast_data="FORECAST",
                critic_type=CriticType.ATTACK_FEASIBILITY,
                assessment=_assessment("attack_feasibility"),
            )
            checkpoint, loaded_path = load_stage1_critic_checkpoint(
                project_root=tmp,
                evidence_pack=_evidence_pack(),
                forecast_data="FORECAST",
                critic_type=CriticType.ATTACK_FEASIBILITY,
            )
            self.assertEqual(path, loaded_path)
            self.assertEqual(checkpoint["assessment"]["critic_type"], "attack_feasibility")

    def test_changed_identity_does_not_fallback_to_old_critic_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            freeze_stage1_critic_checkpoint(
                project_root=tmp,
                evidence_pack=_evidence_pack(),
                forecast_data="FORECAST",
                critic_type=CriticType.ATTACK_FEASIBILITY,
                assessment=_assessment("attack_feasibility"),
            )
            checkpoint, _ = load_stage1_critic_checkpoint(
                project_root=tmp,
                evidence_pack=_evidence_pack(),
                forecast_data="DIFFERENT_FORECAST",
                critic_type=CriticType.ATTACK_FEASIBILITY,
            )
            self.assertIsNone(checkpoint)

    def test_freeze_then_exact_identity_reuses_without_regeneration(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact, path = freeze_stage1_artifact(
                project_root=tmp,
                evidence_pack=_evidence_pack(),
                forecast_data="FORECAST",
                attack_assessment=_assessment("attack_feasibility"),
                defense_assessment=_assessment("defense_robustness"),
            )
            loaded, identity, loaded_path = load_frozen_stage1_artifact(
                project_root=tmp,
                evidence_pack=_evidence_pack(),
                forecast_data="FORECAST",
            )
            self.assertEqual(path, loaded_path)
            self.assertEqual(artifact["artifact_sha256"], loaded["artifact_sha256"])
            self.assertEqual(identity["input_fingerprint"], artifact["input_fingerprint"])

    def test_changed_runtime_profile_creates_a_new_identity_instead_of_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, old_path = freeze_stage1_artifact(
                project_root=tmp,
                evidence_pack=_evidence_pack(),
                forecast_data="FORECAST",
                attack_assessment=_assessment("attack_feasibility"),
                defense_assessment=_assessment("defense_robustness"),
            )
            from Stage1 import artifact as artifact_module

            changed_profile = artifact_module.get_experiment_runtime_profile()
            changed_profile = {**changed_profile, "seed": 999}
            with patch.object(artifact_module, "get_experiment_runtime_profile", return_value=changed_profile):
                loaded, _, new_path = load_frozen_stage1_artifact(
                    project_root=tmp,
                    evidence_pack=_evidence_pack(),
                    forecast_data="FORECAST",
                )
            self.assertIsNone(loaded)
            self.assertNotEqual(old_path, new_path)

    def test_frozen_artifact_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, path = freeze_stage1_artifact(
                project_root=tmp,
                evidence_pack=_evidence_pack(),
                forecast_data="FORECAST",
                attack_assessment=_assessment("attack_feasibility"),
                defense_assessment=_assessment("defense_robustness"),
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["assessments"]["attack_feasibility"]["stance"] = 1
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(Stage1ArtifactError):
                load_frozen_stage1_artifact(
                    project_root=tmp,
                    evidence_pack=_evidence_pack(),
                    forecast_data="FORECAST",
                )

    def test_same_identity_cannot_be_overwritten_with_different_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            freeze_stage1_artifact(
                project_root=tmp,
                evidence_pack=_evidence_pack(),
                forecast_data="FORECAST",
                attack_assessment=_assessment("attack_feasibility"),
                defense_assessment=_assessment("defense_robustness"),
            )
            changed = _assessment("attack_feasibility")
            changed["unresolved_questions"] = ["A materially different unresolved question."]
            with self.assertRaises(Stage1ArtifactError):
                freeze_stage1_artifact(
                    project_root=tmp,
                    evidence_pack=_evidence_pack(),
                    forecast_data="FORECAST",
                    attack_assessment=changed,
                    defense_assessment=_assessment("defense_robustness"),
                )


class Stage1MainCaseSetTests(unittest.TestCase):
    def test_main_case_set_is_frozen_to_seven_unique_cases_and_five_regimes(self):
        validate_main_case_set()
        self.assertEqual(len(MAIN_STAGE1_CASES), 7)
        manifest = main_case_set_manifest()
        self.assertEqual(manifest["population_relation_count"], 303)
        self.assertEqual(len(manifest["cases"]), 7)
        regime_cases = [case for case in MAIN_STAGE1_CASES if case.selection_class == "regime_representative"]
        self.assertEqual(len(regime_cases), 5)


class Stage1GraphTests(unittest.TestCase):
    def test_active_graph_fans_out_before_stage2_debate(self):
        graph = create_graph().get_graph()
        nodes_in_graph = set(graph.nodes)
        edges = {(edge.source, edge.target) for edge in graph.edges}

        self.assertIn(("load_data", "stage1_prepare"), edges)
        self.assertIn(("stage1_prepare", "attack_feasibility_critic"), edges)
        self.assertIn(("stage1_prepare", "defense_robustness_critic"), edges)
        self.assertIn(("attack_feasibility_critic", "stage1_complete"), edges)
        self.assertIn(("defense_robustness_critic", "stage1_complete"), edges)
        self.assertIn(("stage1_complete", "stage2_prepare"), edges)
        self.assertIn(("stage2_prepare", "stage2_debate"), edges)
        self.assertNotIn("attacker", nodes_in_graph)
        self.assertNotIn("defender", nodes_in_graph)
        self.assertNotIn("mediator", nodes_in_graph)


if __name__ == "__main__":
    unittest.main()
