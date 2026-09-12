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
    CriticAssessment,
    CriticType,
    Stage1ValidationError,
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


def _assessment(critic_type: str) -> dict:
    return {
        "case_id": "DDoS__NLP",
        "critic_type": critic_type,
        "stance": 0,
        "confidence": 0.55,
        "evidence_sufficiency": "PARTIAL",
        "claims": [
            {
                "claim_id": "C1",
                "statement": "The available evidence is mixed.",
                "supporting_evidence_ids": ["E001"],
                "contradicting_evidence_ids": ["E002"],
                "status": "MIXED",
            }
        ],
        "unresolved_questions": ["Deployment evidence remains incomplete."],
        "unsupported_specificity_detected": False,
    }


class Stage1SchemaTests(unittest.TestCase):
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
        assessment["claims"][0]["supporting_evidence_ids"] = ["E999"]
        with self.assertRaises(Stage1ValidationError):
            validate_critic_assessment(
                assessment,
                evidence_pack=_evidence_pack(),
                expected_critic_type=CriticType.ATTACK_FEASIBILITY,
            )

    def test_wrong_critic_type_is_rejected(self):
        with self.assertRaises(Stage1ValidationError):
            validate_critic_assessment(
                _assessment("defense_robustness"),
                evidence_pack=_evidence_pack(),
                expected_critic_type=CriticType.ATTACK_FEASIBILITY,
            )

    def test_supported_claim_requires_supporting_evidence(self):
        assessment = _assessment("attack_feasibility")
        claim = assessment["claims"][0]
        claim["status"] = "SUPPORTED"
        claim["supporting_evidence_ids"] = []
        claim["contradicting_evidence_ids"] = []
        with self.assertRaises(Stage1ValidationError):
            validate_critic_assessment(
                assessment,
                evidence_pack=_evidence_pack(),
                expected_critic_type=CriticType.ATTACK_FEASIBILITY,
            )


class _FakeStructuredRunner:
    def __init__(self, result: dict):
        self.result = CriticAssessment.model_validate(result)
        self.messages = None

    async def ainvoke(self, messages):
        self.messages = messages
        return self.result


class _FakeLLM:
    def __init__(self, result: dict):
        self.runner = _FakeStructuredRunner(result)

    def with_structured_output(self, schema, method="json_schema", progress_label=None):
        self.schema = schema
        self.method = method
        self.progress_label = progress_label
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
    def test_prompts_remove_legacy_scenario_dependencies(self):
        self.assertNotIn("{defense_plan}", ATTACK_FEASIBILITY_SYSTEM_PROMPT)
        self.assertNotIn("{attack_plan}", DEFENSE_ROBUSTNESS_SYSTEM_PROMPT)
        self.assertIn("Do **not** design an attack scenario", ATTACK_FEASIBILITY_SYSTEM_PROMPT)
        self.assertIn("Do **not** create a counter-attack plan", DEFENSE_ROBUSTNESS_SYSTEM_PROMPT)
        for prompt in (ATTACK_FEASIBILITY_SYSTEM_PROMPT, DEFENSE_ROBUSTNESS_SYSTEM_PROMPT):
            self.assertIn("# Fixed Internal Review Order", prompt)
            ordered_markers = [
                "Evidence boundary and sufficiency",
                "Forecast-supporting evidence",
                "Forecast-challenging evidence",
                "Alternative explanations",
                "Temporal consistency",
                "Specificity and traceability audit",
                "Residual uncertainty and final judgment",
            ]
            positions = [prompt.index(marker) for marker in ordered_markers]
            self.assertEqual(positions, sorted(positions))

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
        self.assertEqual(config.expected_context_size, 131072)
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
        invalid["claims"][0]["supporting_evidence_ids"] = ["E001", "E002"]
        invalid["claims"][0]["contradicting_evidence_ids"] = ["E002"]
        valid = _assessment("attack_feasibility")
        responses = [
            {"choices": [{"message": {"content": json.dumps(invalid)}}]},
            {"choices": [{"message": {"content": json.dumps(valid)}}]},
        ]
        with patch.object(client, "_post", side_effect=responses) as post:
            parsed = client._invoke_structured(
                [SystemMessage(content="system"), HumanMessage(content="evaluate")],
                CriticAssessment,
            )
        self.assertEqual(parsed.case_id, "DDoS__NLP")
        self.assertEqual(post.call_count, 2)
        repair_messages = post.call_args_list[1].args[0]["messages"]
        self.assertIn("failed deterministic schema/semantic validation", repair_messages[-1]["content"])


class Stage1ArtifactTests(unittest.TestCase):
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
            payload["assessments"]["attack_feasibility"]["confidence"] = 0.99
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
            changed["confidence"] = 0.91
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
    def test_active_graph_fans_out_before_any_debate(self):
        graph = create_graph().get_graph()
        nodes_in_graph = set(graph.nodes)
        edges = {(edge.source, edge.target) for edge in graph.edges}

        self.assertIn(("load_data", "stage1_prepare"), edges)
        self.assertIn(("stage1_prepare", "attack_feasibility_critic"), edges)
        self.assertIn(("stage1_prepare", "defense_robustness_critic"), edges)
        self.assertIn(("attack_feasibility_critic", "stage1_complete"), edges)
        self.assertIn(("defense_robustness_critic", "stage1_complete"), edges)
        self.assertNotIn("attacker", nodes_in_graph)
        self.assertNotIn("defender", nodes_in_graph)
        self.assertNotIn("mediator", nodes_in_graph)


if __name__ == "__main__":
    unittest.main()
