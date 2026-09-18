from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


STAGE2_PARENT = Path(__file__).resolve().parents[2]
if str(STAGE2_PARENT) not in sys.path:
    sys.path.insert(0, str(STAGE2_PARENT))

from Pipeline.graph import create_graph
from Stage1.schema import CriticType
from Stage2 import nodes
from Stage2.artifact import (
    Stage2ArtifactError,
    build_stage2_identity,
    freeze_stage2_artifact,
    freeze_stage2_step_checkpoint,
    load_frozen_stage2_artifact,
    load_stage2_step_checkpoint,
)
from Stage2.prompts import STAGE2_MEDIATOR_SYSTEM_PROMPT, stage2_role_context
from Stage2.schema import (
    DebateRound,
    Stage2ValidationError,
    build_constrained_debate_turn_schema,
    build_constrained_mediator_schema,
    build_constrained_post_assessment_schema,
    flatten_debate_exchanges,
    round_requires_followup,
    surfaced_evidence_ids,
    validate_debate_turn,
    validate_mediator_summary,
    validate_post_assessment,
    validate_stage2_result,
)


def _evidence_pack() -> dict:
    return {
        "case_id": "case__threat__pmt",
        "source_snapshot_id": "snapshot-v1",
        "evidence": [
            {"evidence_id": "E1"},
            {"evidence_id": "E2"},
        ],
    }


def _assessment(critic_type: str) -> dict:
    component = "THREAT_TRAJECTORY" if critic_type == "attack_feasibility" else "PMT_TRAJECTORY"
    return {
        "case_id": "case__threat__pmt",
        "critic_type": critic_type,
        "stance": 0,
        "evidence_sufficiency": "PARTIAL",
        "claims": [
            {
                "claim_id": "C1",
                "statement": "The first supplied record supports one directional interpretation.",
                "evidence_ids": ["E1"],
                "forecast_component": component,
                "forecast_relation": "SUPPORTS_FORECAST",
            },
            {
                "claim_id": "C2",
                "statement": "The second supplied record challenges that directional interpretation.",
                "evidence_ids": ["E2"],
                "forecast_component": component,
                "forecast_relation": "CHALLENGES_FORECAST",
            },
        ],
        "stance_basis": {
            "supporting_claim_ids": ["C1"],
            "challenging_claim_ids": ["C2"],
            "decision": "BALANCED",
            "rationale": "The strongest role-relevant evidence is balanced.",
        },
        "unresolved_questions": [],
        "unsupported_specificity_detected": False,
    }


def _turn(responder: str, round_index: int, response_type: str = "AGREE") -> dict:
    target = "defense_robustness" if responder == "attack_feasibility" else "attack_feasibility"
    evidence_ids = [] if response_type == "INSUFFICIENT_EVIDENCE" else ["E1"]
    return {
        "case_id": "case__threat__pmt",
        "round_index": round_index,
        "responder_critic": responder,
        "target_critic": target,
        "responses": [
            {
                "target_claim_id": "C1",
                "response_type": response_type,
                "evidence_ids": evidence_ids,
                "note": "Grounded response to C1.",
            },
            {
                "target_claim_id": "C2",
                "response_type": "AGREE" if response_type != "INSUFFICIENT_EVIDENCE" else "INSUFFICIENT_EVIDENCE",
                "evidence_ids": ["E2"] if response_type != "INSUFFICIENT_EVIDENCE" else [],
                "note": "Grounded response to C2.",
            },
        ],
    }


def _mediator(
    round_index: int,
    *,
    challenged: bool = False,
    insufficient: bool = False,
    continue_round: bool | None = None,
) -> dict:
    conflicts = []
    gaps = []
    claim_matches = []
    adjudications = []
    next_round_focus = []
    if challenged:
        conflicts = [
            {
                "conflict_id": "K1",
                "attack_claim_ids": ["C1"],
                "defense_claim_ids": ["C1"],
                "evidence_ids": ["E1"],
                "note": "The critics still materially disagree on C1.",
            }
        ]
        claim_matches = [
            {
                "match_id": "M1",
                "attack_claim_ids": ["C1"],
                "defense_claim_ids": ["C1"],
                "status": "UNRESOLVED",
                "evidence_ids": ["E1"],
                "note": "C1 remains contested after comparing the surfaced evidence.",
            }
        ]
        adjudications = [
            {
                "adjudication_id": "A1",
                "attack_claim_ids": ["C1"],
                "defense_claim_ids": ["C1"],
                "outcome": "UNRESOLVED",
                "evidence_ids": ["E1"],
                "rationale": "The surfaced evidence still permits competing interpretations of C1.",
                "required_revision": None,
            }
        ]
    if insufficient:
        gaps = [
            {
                "gap_id": "G1",
                "attack_claim_ids": ["C1", "C2"],
                "defense_claim_ids": ["C1", "C2"],
                "question": "The frozen evidence cannot resolve these claims.",
            }
        ]
        adjudications = [
            {
                "adjudication_id": "A1",
                "attack_claim_ids": ["C1", "C2"],
                "defense_claim_ids": ["C1", "C2"],
                "outcome": "INSUFFICIENT_EVIDENCE",
                "evidence_ids": [],
                "rationale": "The surfaced frozen evidence cannot adjudicate the disputed claims.",
                "required_revision": None,
            }
        ]
    if not challenged and not insufficient:
        claim_matches = [
            {
                "match_id": "M1",
                "attack_claim_ids": ["C1"],
                "defense_claim_ids": ["C1"],
                "status": "RESOLVED",
                "evidence_ids": ["E1"],
                "note": "The surfaced evidence supports a compatible interpretation of C1.",
            }
        ]
        adjudications = [
            {
                "adjudication_id": "A1",
                "attack_claim_ids": ["C1"],
                "defense_claim_ids": ["C1"],
                "outcome": "ACCEPT_BOTH",
                "evidence_ids": ["E1"],
                "rationale": "Both C1 interpretations are adequately grounded by the surfaced evidence.",
                "required_revision": None,
            }
        ]

    if continue_round is None:
        continue_round = challenged and round_index == 1
    if continue_round:
        next_round_focus = [
            {
                "focus_id": "F1",
                "attack_claim_ids": ["C1"],
                "defense_claim_ids": ["C1"],
                "instruction": "Reconsider C1 against the surfaced evidence and address the competing interpretation.",
            }
        ]
    return {
        "case_id": "case__threat__pmt",
        "round_index": round_index,
        "claim_matches": claim_matches,
        "evidence_conflicts": conflicts,
        "evidence_gaps": gaps,
        "adjudications": adjudications,
        "round_action": "CONTINUE" if continue_round else "STOP",
        "next_round_focus": next_round_focus,
        "overall_rationale": "Evidence-bounded adjudication of the current critic exchange.",
    }


def _round(round_index: int, attack_type: str = "AGREE", defense_type: str = "AGREE") -> dict:
    attack_turn = _turn("attack_feasibility", round_index, attack_type)
    defense_turn = _turn("defense_robustness", round_index, defense_type)
    return {
        "round_index": round_index,
        "attack_turn": attack_turn,
        "defense_turn": defense_turn,
        "mediator_summary": _mediator(
            round_index,
            challenged=(attack_type in {"CHALLENGE", "REVISE"} or defense_type in {"CHALLENGE", "REVISE"}),
            insufficient=(attack_type == "INSUFFICIENT_EVIDENCE" or defense_type == "INSUFFICIENT_EVIDENCE"),
        ),
    }


def _result(rounds: list[dict]) -> dict:
    final_mediator = rounds[-1]["mediator_summary"]
    parsed_rounds = [DebateRound.model_validate(item) for item in rounds]
    return {
        "case_id": "case__threat__pmt",
        "attack_pre_assessment": _assessment("attack_feasibility"),
        "defense_pre_assessment": _assessment("defense_robustness"),
        "rounds": rounds,
        "exchanges": [item.model_dump(mode="json") for item in flatten_debate_exchanges(parsed_rounds)],
        "resolved_claims": [
            item for item in final_mediator["claim_matches"] if item["status"] == "RESOLVED"
        ],
        "unresolved_claims": [
            item for item in final_mediator["claim_matches"] if item["status"] == "UNRESOLVED"
        ],
        "evidence_conflicts": final_mediator["evidence_conflicts"],
        "evidence_gaps": final_mediator["evidence_gaps"],
        "final_adjudications": copy.deepcopy(final_mediator["adjudications"]),
        "attack_post_assessment": _assessment("attack_feasibility"),
        "defense_post_assessment": _assessment("defense_robustness"),
        "stance_changes": {"attack_feasibility": 0, "defense_robustness": 0},
    }


def _stage1_artifact() -> dict:
    return {
        "artifact_sha256": "a" * 64,
        "input_fingerprint": "b" * 64,
        "assessments": {
            "attack_feasibility": _assessment("attack_feasibility"),
            "defense_robustness": _assessment("defense_robustness"),
        },
    }


class _FakeStructuredRunner:
    def __init__(self, owner, schema, response_schema, progress_label, semantic_validator):
        self.owner = owner
        self.schema = schema
        self.response_schema = response_schema
        self.progress_label = progress_label
        self.semantic_validator = semantic_validator

    async def ainvoke(self, messages):
        self.owner.calls.append(
            {
                "schema": self.schema.__name__,
                "progress_label": self.progress_label,
                "messages": messages,
            }
        )
        if self.schema.__name__ == "CriticDebateTurn":
            responder = self.response_schema["properties"]["responder_critic"]["const"]
            round_index = self.response_schema["properties"]["round_index"]["const"]
            response_type = (
                "CHALLENGE"
                if self.owner.challenge_round1 and responder == "attack_feasibility" and round_index == 1
                else "AGREE"
            )
            payload = _turn(responder, round_index, response_type)
        elif self.schema.__name__ == "MediatorSummary":
            round_index = self.response_schema["properties"]["round_index"]["const"]
            payload = _mediator(
                round_index,
                challenged=(self.owner.challenge_round1 and round_index == 1),
            )
        elif self.schema.__name__ == "CriticAssessment":
            critic_type = self.response_schema["properties"]["critic_type"]["const"]
            payload = _assessment(critic_type)
        else:  # pragma: no cover - guard against future unhandled Stage 2 schemas.
            raise AssertionError(f"Unexpected structured schema: {self.schema.__name__}")

        parsed = self.schema.model_validate(payload)
        return self.semantic_validator(parsed) if self.semantic_validator else parsed


class _FakeStage2LLM:
    def __init__(self, *, challenge_round1: bool = False):
        self.challenge_round1 = challenge_round1
        self.calls: list[dict] = []

    def with_structured_output(
        self,
        schema,
        *,
        method="json_schema",
        progress_label=None,
        semantic_validator=None,
        response_schema=None,
    ):
        self_method = method
        if self_method != "json_schema":
            raise AssertionError("Stage 2 must use JSON-schema constrained output")
        return _FakeStructuredRunner(
            self,
            schema,
            response_schema,
            progress_label,
            semantic_validator,
        )


class Stage2SchemaTests(unittest.TestCase):
    def test_constrained_turn_schema_binds_exact_claim_and_evidence_ids(self):
        schema = build_constrained_debate_turn_schema(
            evidence_pack=_evidence_pack(),
            opponent_assessment=_assessment("defense_robustness"),
            expected_responder=CriticType.ATTACK_FEASIBILITY,
            round_index=1,
        )
        self.assertEqual(schema["properties"]["case_id"]["const"], "case__threat__pmt")
        self.assertEqual(schema["properties"]["round_index"]["const"], 1)
        response = schema["$defs"]["DebateResponseItem"]["properties"]
        self.assertEqual(response["target_claim_id"]["enum"], ["C1", "C2"])
        self.assertEqual(response["evidence_ids"]["items"]["enum"], ["E1", "E2"])

    def test_turn_must_review_every_opponent_claim_exactly_once(self):
        turn = _turn("attack_feasibility", 1)
        turn["responses"] = turn["responses"][:1]
        with self.assertRaisesRegex(Stage2ValidationError, "every opponent claim"):
            validate_debate_turn(
                turn,
                evidence_pack=_evidence_pack(),
                opponent_assessment=_assessment("defense_robustness"),
                expected_responder=CriticType.ATTACK_FEASIBILITY,
                round_index=1,
            )

    def test_constrained_mediator_schema_binds_both_claim_sets_and_evidence_ids(self):
        schema = build_constrained_mediator_schema(
            evidence_pack=_evidence_pack(),
            attack_assessment=_assessment("attack_feasibility"),
            defense_assessment=_assessment("defense_robustness"),
            round_index=1,
        )
        self.assertEqual(schema["properties"]["case_id"]["const"], "case__threat__pmt")
        conflict = schema["$defs"]["EvidenceConflict"]["properties"]
        self.assertEqual(conflict["attack_claim_ids"]["items"]["enum"], ["C1", "C2"])
        self.assertEqual(conflict["defense_claim_ids"]["items"]["enum"], ["C1", "C2"])
        self.assertEqual(conflict["evidence_ids"]["items"]["enum"], ["E1", "E2"])

        adjudication = schema["$defs"]["MediatorAdjudication"]
        self.assertIn("attack_claim_ids", adjudication["required"])
        self.assertIn("defense_claim_ids", adjudication["required"])
        self.assertIn("required_revision", adjudication["required"])
        revision_schema = adjudication["properties"]["required_revision"]
        self.assertIn({"type": "string", "minLength": 1}, revision_schema["anyOf"])
        self.assertIn({"type": "null"}, revision_schema["anyOf"])

    def test_non_gap_response_requires_grounding_evidence(self):
        turn = _turn("attack_feasibility", 1)
        turn["responses"][0]["evidence_ids"] = []
        with self.assertRaises(Stage2ValidationError):
            validate_debate_turn(
                turn,
                evidence_pack=_evidence_pack(),
                opponent_assessment=_assessment("defense_robustness"),
                expected_responder=CriticType.ATTACK_FEASIBILITY,
                round_index=1,
            )

    def test_agree_must_recite_target_claim_grounding(self):
        evidence_pack = _evidence_pack()
        turn = _turn("attack_feasibility", 1)
        turn["responses"][0]["evidence_ids"] = ["E2"]
        with self.assertRaisesRegex(Stage2ValidationError, "must re-cite at least one evidence ID"):
            validate_debate_turn(
                turn,
                evidence_pack=evidence_pack,
                opponent_assessment=_assessment("defense_robustness"),
                expected_responder=CriticType.ATTACK_FEASIBILITY,
                round_index=1,
            )

    def test_insufficient_evidence_response_must_not_cite_evidence(self):
        turn = _turn("attack_feasibility", 1, "INSUFFICIENT_EVIDENCE")
        turn["responses"][0]["evidence_ids"] = ["E1"]
        with self.assertRaises(Stage2ValidationError):
            validate_debate_turn(
                turn,
                evidence_pack=_evidence_pack(),
                opponent_assessment=_assessment("defense_robustness"),
                expected_responder=CriticType.ATTACK_FEASIBILITY,
                round_index=1,
            )

    def test_critic_insufficient_label_does_not_mechanically_force_mediator_gap(self):
        attack_turn = _turn("attack_feasibility", 1, "INSUFFICIENT_EVIDENCE")
        defense_turn = _turn("defense_robustness", 1)
        summary = _mediator(1)
        summary["claim_matches"] = []
        summary["adjudications"] = [
            {
                "adjudication_id": "A_OVERRIDE",
                "attack_claim_ids": [],
                "defense_claim_ids": ["C1", "C2"],
                "outcome": "ACCEPT_DEFENSE",
                "evidence_ids": ["E1", "E2"],
                "rationale": "The surfaced evidence is sufficient to accept the Defense claims despite the Attack critic abstention.",
                "required_revision": None,
            }
        ]
        parsed = validate_mediator_summary(
            summary,
            evidence_pack=_evidence_pack(),
            attack_assessment=_assessment("attack_feasibility"),
            defense_assessment=_assessment("defense_robustness"),
            attack_turn=attack_turn,
            defense_turn=defense_turn,
            round_index=1,
        )
        self.assertEqual(parsed.adjudications[0].outcome.value, "ACCEPT_DEFENSE")
        self.assertEqual(parsed.evidence_gaps, [])

    def test_mediator_insufficient_adjudication_requires_evidence_gap(self):
        summary = _mediator(1, insufficient=True)
        summary["evidence_gaps"] = []
        with self.assertRaisesRegex(Stage2ValidationError, "must be represented by an evidence gap with the same"):
            validate_mediator_summary(
                summary,
                evidence_pack=_evidence_pack(),
                attack_assessment=_assessment("attack_feasibility"),
                defense_assessment=_assessment("defense_robustness"),
                attack_turn=_turn("attack_feasibility", 1, "INSUFFICIENT_EVIDENCE"),
                defense_turn=_turn("defense_robustness", 1),
                round_index=1,
            )

    def test_mediator_cannot_drop_a_challenged_claim(self):
        attack_turn = _turn("attack_feasibility", 1, "CHALLENGE")
        defense_turn = _turn("defense_robustness", 1)
        summary = _mediator(1)
        summary["claim_matches"] = []
        summary["adjudications"] = [
            {
                "adjudication_id": "A_OTHER",
                "attack_claim_ids": ["C2"],
                "defense_claim_ids": ["C2"],
                "outcome": "ACCEPT_BOTH",
                "evidence_ids": ["E2"],
                "rationale": "Only C2 was adjudicated.",
                "required_revision": None,
            }
        ]
        with self.assertRaisesRegex(Stage2ValidationError, "omitted a challenged/revised/insufficient Defense claim"):
            validate_mediator_summary(
                summary,
                evidence_pack=_evidence_pack(),
                attack_assessment=_assessment("attack_feasibility"),
                defense_assessment=_assessment("defense_robustness"),
                attack_turn=attack_turn,
                defense_turn=defense_turn,
                round_index=1,
            )

    def test_evidence_gap_cannot_substitute_for_required_adjudication(self):
        attack_turn = _turn("attack_feasibility", 1, "CHALLENGE")
        defense_turn = _turn("defense_robustness", 1)
        gap_only = _mediator(1, insufficient=True)
        gap_only["adjudications"] = [
            {
                "adjudication_id": "A_OTHER",
                "attack_claim_ids": ["C2"],
                "defense_claim_ids": ["C2"],
                "outcome": "ACCEPT_BOTH",
                "evidence_ids": ["E2"],
                "rationale": "The challenged C1 dispute was not adjudicated.",
                "required_revision": None,
            }
        ]
        with self.assertRaisesRegex(
            Stage2ValidationError,
            "omitted a challenged/revised/insufficient Defense claim",
        ):
            validate_mediator_summary(
                gap_only,
                evidence_pack=_evidence_pack(),
                attack_assessment=_assessment("attack_feasibility"),
                defense_assessment=_assessment("defense_robustness"),
                attack_turn=attack_turn,
                defense_turn=defense_turn,
                round_index=1,
            )

    def test_mediator_may_identify_conflict_even_when_critics_agree(self):
        summary = _mediator(1)
        summary["evidence_conflicts"] = [
            {
                "conflict_id": "K_SPURIOUS",
                "attack_claim_ids": ["C1"],
                "defense_claim_ids": ["C1"],
                "evidence_ids": ["E1"],
                "note": "This conflict is spurious because both current responses agree.",
            }
        ]
        parsed = validate_mediator_summary(
            summary,
            evidence_pack=_evidence_pack(),
            attack_assessment=_assessment("attack_feasibility"),
            defense_assessment=_assessment("defense_robustness"),
            attack_turn=_turn("attack_feasibility", 1),
            defense_turn=_turn("defense_robustness", 1),
            round_index=1,
        )
        self.assertEqual(len(parsed.evidence_conflicts), 1)

    def test_mediator_may_identify_gap_even_when_critics_did_not_abstain(self):
        summary = _mediator(1)
        summary["evidence_gaps"] = [
            {
                "gap_id": "G_SPURIOUS",
                "attack_claim_ids": ["C1"],
                "defense_claim_ids": ["C1"],
                "question": "This gap is spurious because neither current response is insufficient.",
            }
        ]
        parsed = validate_mediator_summary(
            summary,
            evidence_pack=_evidence_pack(),
            attack_assessment=_assessment("attack_feasibility"),
            defense_assessment=_assessment("defense_robustness"),
            attack_turn=_turn("attack_feasibility", 1),
            defense_turn=_turn("defense_robustness", 1),
            round_index=1,
        )
        self.assertEqual(len(parsed.evidence_gaps), 1)

    def test_mediator_match_status_follows_mediator_adjudication_not_critic_labels(self):
        attack_turn = _turn("attack_feasibility", 1, "CHALLENGE")
        defense_turn = _turn("defense_robustness", 1)
        summary = _mediator(1, challenged=True)
        summary["claim_matches"] = [
            {
                "match_id": "M1",
                "attack_claim_ids": ["C1"],
                "defense_claim_ids": ["C1"],
                "status": "RESOLVED",
                "evidence_ids": ["E1"],
                "note": "The Mediator rejects the critic challenge after comparing the surfaced evidence.",
            }
        ]
        summary["adjudications"] = [
            {
                "adjudication_id": "A1",
                "attack_claim_ids": ["C1"],
                "defense_claim_ids": ["C1"],
                "outcome": "ACCEPT_BOTH",
                "evidence_ids": ["E1"],
                "rationale": "The challenge does not outweigh the grounded compatible interpretation.",
                "required_revision": None,
            }
        ]
        summary["round_action"] = "STOP"
        summary["next_round_focus"] = []
        parsed = validate_mediator_summary(
            summary,
            evidence_pack=_evidence_pack(),
            attack_assessment=_assessment("attack_feasibility"),
            defense_assessment=_assessment("defense_robustness"),
            attack_turn=attack_turn,
            defense_turn=defense_turn,
            round_index=1,
        )
        self.assertEqual(parsed.claim_matches[0].status.value, "RESOLVED")

        agree_summary = _mediator(1)
        agree_summary["claim_matches"] = [
            {
                "match_id": "M1",
                "attack_claim_ids": ["C1"],
                "defense_claim_ids": ["C1"],
                "status": "UNRESOLVED",
                "evidence_ids": ["E1"],
                "note": "The Mediator finds the apparent agreement unsupported by the surfaced evidence.",
            }
        ]
        agree_summary["adjudications"] = [
            {
                "adjudication_id": "A1",
                "attack_claim_ids": ["C1"],
                "defense_claim_ids": ["C1"],
                "outcome": "UNRESOLVED",
                "evidence_ids": ["E1"],
                "rationale": "The surfaced evidence does not justify treating the apparent agreement as resolved.",
                "required_revision": None,
            }
        ]
        parsed = validate_mediator_summary(
            agree_summary,
            evidence_pack=_evidence_pack(),
            attack_assessment=_assessment("attack_feasibility"),
            defense_assessment=_assessment("defense_robustness"),
            attack_turn=_turn("attack_feasibility", 1),
            defense_turn=_turn("defense_robustness", 1),
            round_index=1,
        )
        self.assertEqual(parsed.claim_matches[0].status.value, "UNRESOLVED")

    def test_claim_match_requires_full_coverage_by_one_adjudication(self):
        summary = _mediator(1)
        summary["adjudications"] = [
            {
                "adjudication_id": "A_PARTIAL",
                "attack_claim_ids": ["C1"],
                "defense_claim_ids": [],
                "outcome": "ACCEPT_ATTACK",
                "evidence_ids": ["E1"],
                "rationale": "Only the Attack side of the matched dispute was adjudicated.",
                "required_revision": None,
            }
        ]
        with self.assertRaisesRegex(Stage2ValidationError, "map to exactly one adjudication"):
            validate_mediator_summary(
                summary,
                evidence_pack=_evidence_pack(),
                attack_assessment=_assessment("attack_feasibility"),
                defense_assessment=_assessment("defense_robustness"),
                attack_turn=_turn("attack_feasibility", 1),
                defense_turn=_turn("defense_robustness", 1),
                round_index=1,
            )

    def test_exact_same_dispute_unit_cannot_receive_multiple_adjudications(self):
        summary = _mediator(1)
        summary["adjudications"].append(
            {
                "adjudication_id": "A_CONTRADICT",
                "attack_claim_ids": ["C1"],
                "defense_claim_ids": ["C1"],
                "outcome": "REJECT_BOTH",
                "evidence_ids": ["E1"],
                "rationale": "A second judgment over the exact same dispute unit must be rejected.",
                "required_revision": None,
            }
        )
        with self.assertRaisesRegex(Stage2ValidationError, "exact same Attack/Defense claim set"):
            validate_mediator_summary(
                summary,
                evidence_pack=_evidence_pack(),
                attack_assessment=_assessment("attack_feasibility"),
                defense_assessment=_assessment("defense_robustness"),
                attack_turn=_turn("attack_feasibility", 1),
                defense_turn=_turn("defense_robustness", 1),
                round_index=1,
            )

    def test_insufficient_adjudication_gap_must_cover_all_referenced_claims(self):
        summary = _mediator(1, insufficient=True)
        summary["evidence_gaps"][0]["defense_claim_ids"] = ["C1"]
        with self.assertRaisesRegex(Stage2ValidationError, "same referenced claim set"):
            validate_mediator_summary(
                summary,
                evidence_pack=_evidence_pack(),
                attack_assessment=_assessment("attack_feasibility"),
                defense_assessment=_assessment("defense_robustness"),
                attack_turn=_turn("attack_feasibility", 1, "INSUFFICIENT_EVIDENCE"),
                defense_turn=_turn("defense_robustness", 1),
                round_index=1,
            )

    def test_next_round_focus_cannot_smuggle_unrelated_claims(self):
        summary = _mediator(1, challenged=True, continue_round=True)
        summary["next_round_focus"][0]["defense_claim_ids"].append("C2")
        with self.assertRaisesRegex(Stage2ValidationError, "stay wholly within one"):
            validate_mediator_summary(
                summary,
                evidence_pack=_evidence_pack(),
                attack_assessment=_assessment("attack_feasibility"),
                defense_assessment=_assessment("defense_robustness"),
                attack_turn=_turn("attack_feasibility", 1, "CHALLENGE"),
                defense_turn=_turn("defense_robustness", 1),
                round_index=1,
            )

    def test_same_claim_may_receive_multiple_distinct_next_round_focus_instructions(self):
        summary = _mediator(1, challenged=True, continue_round=True)
        summary["next_round_focus"].append(
            {
                "focus_id": "F2",
                "attack_claim_ids": ["C1"],
                "defense_claim_ids": ["C1"],
                "instruction": "Check a second distinct aspect of the same unresolved dispute.",
            }
        )
        parsed = validate_mediator_summary(
            summary,
            evidence_pack=_evidence_pack(),
            attack_assessment=_assessment("attack_feasibility"),
            defense_assessment=_assessment("defense_robustness"),
            attack_turn=_turn("attack_feasibility", 1, "CHALLENGE"),
            defense_turn=_turn("defense_robustness", 1),
            round_index=1,
        )
        self.assertEqual(len(parsed.next_round_focus), 2)

    def test_mediator_cannot_introduce_unused_stage0_evidence(self):
        evidence_pack = _evidence_pack()
        evidence_pack["evidence"].append({"evidence_id": "E_UNUSED"})
        attack_turn = _turn("attack_feasibility", 1, "CHALLENGE")
        defense_turn = _turn("defense_robustness", 1)
        summary = _mediator(1, challenged=True)
        summary["evidence_conflicts"][0]["evidence_ids"] = ["E_UNUSED"]
        with self.assertRaisesRegex(Stage2ValidationError, "Mediator may judge only from evidence"):
            validate_mediator_summary(
                summary,
                evidence_pack=evidence_pack,
                attack_assessment=_assessment("attack_feasibility"),
                defense_assessment=_assessment("defense_robustness"),
                attack_turn=attack_turn,
                defense_turn=defense_turn,
                round_index=1,
            )

    def test_one_sided_challenge_does_not_require_fake_cross_critic_match(self):
        attack_turn = _turn("attack_feasibility", 1, "CHALLENGE")
        defense_turn = _turn("defense_robustness", 1)
        summary = _mediator(1)
        summary["claim_matches"] = []
        summary["evidence_conflicts"] = [
            {
                "conflict_id": "K_ONE_SIDED",
                "attack_claim_ids": [],
                "defense_claim_ids": ["C1"],
                "evidence_ids": ["E1"],
                "note": "Attack challenges Defense C1 without inventing a matching Attack pre-claim.",
            }
        ]
        summary["adjudications"] = [
            {
                "adjudication_id": "A_ONE_SIDED",
                "attack_claim_ids": [],
                "defense_claim_ids": ["C1"],
                "outcome": "REJECT_DEFENSE",
                "evidence_ids": ["E1"],
                "rationale": "The surfaced evidence does not adequately support Defense C1 as written.",
                "required_revision": None,
            }
        ]
        summary["round_action"] = "STOP"
        summary["next_round_focus"] = []
        parsed = validate_mediator_summary(
            summary,
            evidence_pack=_evidence_pack(),
            attack_assessment=_assessment("attack_feasibility"),
            defense_assessment=_assessment("defense_robustness"),
            attack_turn=attack_turn,
            defense_turn=defense_turn,
            round_index=1,
        )
        self.assertEqual(parsed.evidence_conflicts[0].attack_claim_ids, [])
        self.assertEqual(parsed.evidence_conflicts[0].defense_claim_ids, ["C1"])
        self.assertEqual(parsed.adjudications[0].outcome.value, "REJECT_DEFENSE")

    def test_mediator_evidence_must_be_local_to_referenced_claims(self):
        attack_turn = _turn("attack_feasibility", 1, "CHALLENGE")
        defense_turn = _turn("defense_robustness", 1)
        summary = _mediator(1, challenged=True)
        # E2 is globally surfaced by C2, but the conflict below references only
        # C1 on both sides. Global surfacing must not authorize cross-claim
        # evidence reassignment by the Mediator.
        summary["evidence_conflicts"][0]["evidence_ids"] = ["E2"]
        with self.assertRaisesRegex(Stage2ValidationError, "not surfaced for the claim IDs"):
            validate_mediator_summary(
                summary,
                evidence_pack=_evidence_pack(),
                attack_assessment=_assessment("attack_feasibility"),
                defense_assessment=_assessment("defense_robustness"),
                attack_turn=attack_turn,
                defense_turn=defense_turn,
                round_index=1,
            )

    def test_non_gap_adjudication_must_cite_each_referenced_claim(self):
        summary = _mediator(1)
        summary["adjudications"] = [
            {
                "adjudication_id": "A_MULTI",
                "attack_claim_ids": ["C1", "C2"],
                "defense_claim_ids": ["C1"],
                "outcome": "ACCEPT_BOTH",
                "evidence_ids": ["E1"],
                "rationale": "E1 does not cover Attack C2, so this judgment is incompletely grounded.",
                "required_revision": None,
            }
        ]
        summary["claim_matches"] = []
        with self.assertRaisesRegex(Stage2ValidationError, "every referenced claim"):
            validate_mediator_summary(
                summary,
                evidence_pack=_evidence_pack(),
                attack_assessment=_assessment("attack_feasibility"),
                defense_assessment=_assessment("defense_robustness"),
                attack_turn=_turn("attack_feasibility", 1),
                defense_turn=_turn("defense_robustness", 1),
                round_index=1,
            )

    def test_runtime_mediator_schema_excludes_unused_stage0_evidence(self):
        evidence_pack = _evidence_pack()
        evidence_pack["evidence"].append({"evidence_id": "E_UNUSED"})
        schema = build_constrained_mediator_schema(
            evidence_pack=evidence_pack,
            attack_assessment=_assessment("attack_feasibility"),
            defense_assessment=_assessment("defense_robustness"),
            round_index=1,
            attack_turn=_turn("attack_feasibility", 1),
            defense_turn=_turn("defense_robustness", 1),
        )
        evidence_enum = schema["$defs"]["EvidenceConflict"]["properties"]["evidence_ids"]["items"]["enum"]
        self.assertEqual(evidence_enum, ["E1", "E2"])

    def test_round2_mediator_schema_can_reuse_evidence_surfaced_by_round1_critic(self):
        evidence_pack = _evidence_pack()
        evidence_pack["evidence"].append({"evidence_id": "E_PREVIOUS"})
        previous = _round(1, "CHALLENGE", "AGREE")
        previous["attack_turn"]["responses"][0]["evidence_ids"] = ["E_PREVIOUS"]
        schema = build_constrained_mediator_schema(
            evidence_pack=evidence_pack,
            attack_assessment=_assessment("attack_feasibility"),
            defense_assessment=_assessment("defense_robustness"),
            round_index=2,
            attack_turn=_turn("attack_feasibility", 2),
            defense_turn=_turn("defense_robustness", 2),
            previous_rounds=[previous],
        )
        evidence_enum = schema["$defs"]["EvidenceConflict"]["properties"]["evidence_ids"]["items"]["enum"]
        self.assertEqual(evidence_enum, ["E1", "E2", "E_PREVIOUS"])

    def test_round2_routing_is_controlled_by_mediator_action(self):
        self.assertFalse(round_requires_followup(DebateRound.model_validate(_round(1))))
        self.assertFalse(
            round_requires_followup(DebateRound.model_validate(_round(1, "INSUFFICIENT_EVIDENCE", "AGREE")))
        )
        self.assertTrue(round_requires_followup(DebateRound.model_validate(_round(1, "CHALLENGE", "AGREE"))))
        self.assertTrue(round_requires_followup(DebateRound.model_validate(_round(1, "REVISE", "AGREE"))))

        challenged_but_stopped = _round(1, "CHALLENGE", "AGREE")
        challenged_but_stopped["mediator_summary"] = _mediator(
            1,
            challenged=True,
            continue_round=False,
        )
        self.assertFalse(round_requires_followup(DebateRound.model_validate(challenged_but_stopped)))

        agreed_but_continued = _round(1)
        agreed_but_continued["mediator_summary"] = _mediator(1, challenged=True, continue_round=True)
        self.assertTrue(round_requires_followup(DebateRound.model_validate(agreed_but_continued)))

    def test_round2_cannot_introduce_evidence_not_surfaced_before_round2(self):
        evidence_pack = _evidence_pack()
        evidence_pack["evidence"].append({"evidence_id": "E3"})
        round1 = DebateRound.model_validate(_round(1, "CHALLENGE", "AGREE"))
        allowed = surfaced_evidence_ids(
            attack_assessment=_assessment("attack_feasibility"),
            defense_assessment=_assessment("defense_robustness"),
            previous_rounds=[round1],
        )
        self.assertEqual(allowed, ["E1", "E2"])
        schema = build_constrained_debate_turn_schema(
            evidence_pack=evidence_pack,
            opponent_assessment=_assessment("defense_robustness"),
            expected_responder=CriticType.ATTACK_FEASIBILITY,
            round_index=2,
            allowed_evidence_ids=allowed,
        )
        evidence_enum = schema["$defs"]["DebateResponseItem"]["properties"]["evidence_ids"]["items"]["enum"]
        self.assertEqual(evidence_enum, ["E1", "E2"])
        round2_turn = _turn("attack_feasibility", 2)
        round2_turn["responses"][0]["evidence_ids"] = ["E3"]
        with self.assertRaisesRegex(Stage2ValidationError, "outside the allowed evidence set"):
            validate_debate_turn(
                round2_turn,
                evidence_pack=evidence_pack,
                opponent_assessment=_assessment("defense_robustness"),
                expected_responder=CriticType.ATTACK_FEASIBILITY,
                round_index=2,
                allowed_evidence_ids=allowed,
            )

    def test_post_assessment_cannot_introduce_unsurfaced_evidence(self):
        evidence_pack = _evidence_pack()
        evidence_pack["evidence"].append({"evidence_id": "E3"})
        rounds = [DebateRound.model_validate(_round(1))]
        schema = build_constrained_post_assessment_schema(
            evidence_pack=evidence_pack,
            expected_critic_type=CriticType.ATTACK_FEASIBILITY,
            attack_pre_assessment=_assessment("attack_feasibility"),
            defense_pre_assessment=_assessment("defense_robustness"),
            rounds=rounds,
        )
        evidence_enum = schema["$defs"]["ClaimAssessment"]["properties"]["evidence_ids"]["items"]["enum"]
        self.assertEqual(evidence_enum, ["E1", "E2"])
        post = _assessment("attack_feasibility")
        post["claims"][0]["evidence_ids"] = ["E3"]
        with self.assertRaisesRegex(Stage2ValidationError, "never surfaced"):
            validate_post_assessment(
                post,
                evidence_pack=evidence_pack,
                expected_critic_type=CriticType.ATTACK_FEASIBILITY,
                attack_pre_assessment=_assessment("attack_feasibility"),
                defense_pre_assessment=_assessment("defense_robustness"),
                rounds=rounds,
            )

    def test_one_round_result_rejects_unfollowed_mediator_continue(self):
        with self.assertRaisesRegex(Stage2ValidationError, "fixed second round is required"):
            validate_stage2_result(
                _result([_round(1, "CHALLENGE", "AGREE")]),
                evidence_pack=_evidence_pack(),
                attack_pre_assessment=_assessment("attack_feasibility"),
                defense_pre_assessment=_assessment("defense_robustness"),
            )

    def test_two_round_result_is_valid_only_when_round1_mediator_requests_followup(self):
        parsed = validate_stage2_result(
            _result([_round(1, "CHALLENGE", "AGREE"), _round(2)]),
            evidence_pack=_evidence_pack(),
            attack_pre_assessment=_assessment("attack_feasibility"),
            defense_pre_assessment=_assessment("defense_robustness"),
        )
        self.assertEqual(len(parsed.rounds), 2)
        with self.assertRaisesRegex(Stage2ValidationError, "Round 2 is allowed only"):
            validate_stage2_result(
                _result([_round(1), _round(2)]),
                evidence_pack=_evidence_pack(),
                attack_pre_assessment=_assessment("attack_feasibility"),
                defense_pre_assessment=_assessment("defense_robustness"),
            )

    def test_pre_assessments_are_immutable(self):
        result = _result([_round(1)])
        result["attack_pre_assessment"]["unresolved_questions"] = ["mutated"]
        with self.assertRaisesRegex(Stage2ValidationError, "pre-assessment was modified"):
            validate_stage2_result(
                result,
                evidence_pack=_evidence_pack(),
                attack_pre_assessment=_assessment("attack_feasibility"),
                defense_pre_assessment=_assessment("defense_robustness"),
            )

    def test_flattened_exchanges_are_part_of_the_result_contract(self):
        result = _result([_round(1)])
        result["exchanges"][0]["note"] = "tampered"
        with self.assertRaisesRegex(Stage2ValidationError, "exact programmatic flattening"):
            validate_stage2_result(
                result,
                evidence_pack=_evidence_pack(),
                attack_pre_assessment=_assessment("attack_feasibility"),
                defense_pre_assessment=_assessment("defense_robustness"),
            )

    def test_final_adjudications_are_exact_final_mediator_snapshot(self):
        result = _result([_round(1)])
        result["final_adjudications"][0]["rationale"] = "tampered"
        with self.assertRaisesRegex(Stage2ValidationError, "complete final Mediator adjudication snapshot"):
            validate_stage2_result(
                result,
                evidence_pack=_evidence_pack(),
                attack_pre_assessment=_assessment("attack_feasibility"),
                defense_pre_assessment=_assessment("defense_robustness"),
            )

    def test_mediator_prompt_defines_evidence_bounded_adjudicator_role(self):
        self.assertIn("evidence-bounded adjudicator", STAGE2_MEDIATOR_SYSTEM_PROMPT)
        self.assertIn("not treated as external", STAGE2_MEDIATOR_SYSTEM_PROMPT)
        self.assertIn("ACCEPT_ATTACK", STAGE2_MEDIATOR_SYSTEM_PROMPT)
        self.assertIn("REJECT_DEFENSE", STAGE2_MEDIATOR_SYSTEM_PROMPT)
        self.assertIn("Decide `round_action` yourself", STAGE2_MEDIATOR_SYSTEM_PROMPT)
        self.assertIn("risk percentages", STAGE2_MEDIATOR_SYSTEM_PROMPT)

    def test_stage2_role_context_does_not_reuse_stage1_independence_or_output_instructions(self):
        attack = stage2_role_context(critic_type="attack_feasibility", forecast_data="FORECAST")
        defense = stage2_role_context(critic_type="defense_robustness", forecast_data="FORECAST")
        for prompt in (attack, defense):
            self.assertNotIn("independent initial assessment", prompt)
            self.assertNotIn("Do not assume, request, or infer", prompt)
            self.assertNotIn("Return only the structured `CriticAssessment`", prompt)
            self.assertIn("FORECAST", prompt)


class Stage2ArtifactTests(unittest.TestCase):
    def test_identity_reuses_exact_stage1_runtime_profile(self):
        from Stage1.runtime import get_experiment_runtime_profile

        identity = build_stage2_identity(
            evidence_pack=_evidence_pack(),
            forecast_data="FORECAST",
            stage1_artifact=_stage1_artifact(),
        )
        self.assertEqual(identity["runtime"], get_experiment_runtime_profile())
        self.assertEqual(identity["max_rounds"], 2)

    def test_step_checkpoint_reuses_exact_payload_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            identity = build_stage2_identity(
                evidence_pack=_evidence_pack(),
                forecast_data="FORECAST",
                stage1_artifact=_stage1_artifact(),
            )
            dependency = {"previous_rounds": [], "responder": "attack_feasibility"}
            path = freeze_stage2_step_checkpoint(
                project_root=tmp,
                identity=identity,
                step_name="round1_attack",
                payload=_turn("attack_feasibility", 1),
                dependency_payload=dependency,
            )
            checkpoint, loaded_path = load_stage2_step_checkpoint(
                project_root=tmp,
                identity=identity,
                step_name="round1_attack",
                dependency_payload=dependency,
            )
            self.assertEqual(path, loaded_path)
            self.assertEqual(checkpoint["payload"]["round_index"], 1)
            raw = json.loads(path.read_text(encoding="utf-8"))
            raw["payload"]["round_index"] = 2
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(Stage2ArtifactError):
                load_stage2_step_checkpoint(
                    project_root=tmp,
                    identity=identity,
                    step_name="round1_attack",
                    dependency_payload=dependency,
                )

    def test_step_checkpoint_invalidates_changed_upstream_dependency(self):
        with tempfile.TemporaryDirectory() as tmp:
            identity = build_stage2_identity(
                evidence_pack=_evidence_pack(),
                forecast_data="FORECAST",
                stage1_artifact=_stage1_artifact(),
            )
            path = freeze_stage2_step_checkpoint(
                project_root=tmp,
                identity=identity,
                step_name="round2_mediator",
                payload=_mediator(2),
                dependency_payload={"round1": "old"},
            )
            checkpoint, loaded_path = load_stage2_step_checkpoint(
                project_root=tmp,
                identity=identity,
                step_name="round2_mediator",
                dependency_payload={"round1": "regenerated-differently"},
            )
            self.assertIsNone(checkpoint)
            self.assertEqual(path, loaded_path)
            self.assertFalse(path.exists())

            regenerated = freeze_stage2_step_checkpoint(
                project_root=tmp,
                identity=identity,
                step_name="round2_mediator",
                payload=_mediator(2),
                dependency_payload={"round1": "regenerated-differently"},
            )
            self.assertTrue(regenerated.exists())

    def test_freeze_then_reuse_final_stage2_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact, path = freeze_stage2_artifact(
                project_root=tmp,
                evidence_pack=_evidence_pack(),
                forecast_data="FORECAST",
                stage1_artifact=_stage1_artifact(),
                attack_pre_assessment=_assessment("attack_feasibility"),
                defense_pre_assessment=_assessment("defense_robustness"),
                debate_result=_result([_round(1)]),
            )
            loaded, identity, loaded_path = load_frozen_stage2_artifact(
                project_root=tmp,
                evidence_pack=_evidence_pack(),
                forecast_data="FORECAST",
                stage1_artifact=_stage1_artifact(),
                attack_pre_assessment=_assessment("attack_feasibility"),
                defense_pre_assessment=_assessment("defense_robustness"),
            )
            self.assertEqual(path, loaded_path)
            self.assertEqual(loaded["artifact_sha256"], artifact["artifact_sha256"])
            self.assertEqual(identity["input_fingerprint"], artifact["input_fingerprint"])

    def test_artifact_layer_rejects_pre_assessment_not_bound_to_stage1_artifact(self):
        mutated_attack = _assessment("attack_feasibility")
        mutated_attack["unresolved_questions"] = ["not the frozen Stage 1 pre-assessment"]
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(Stage2ArtifactError, "does not match the bound frozen Stage 1 artifact"):
                freeze_stage2_artifact(
                    project_root=tmp,
                    evidence_pack=_evidence_pack(),
                    forecast_data="FORECAST",
                    stage1_artifact=_stage1_artifact(),
                    attack_pre_assessment=mutated_attack,
                    defense_pre_assessment=_assessment("defense_robustness"),
                    debate_result=_result([_round(1)]),
                )


class Stage2NodeAndGraphTests(unittest.TestCase):
    def test_prepare_reuses_final_artifact_without_llm(self):
        state = {
            "stage1_complete": True,
            "forecast_data": "FORECAST",
            "evidence_pack": _evidence_pack(),
            "attack_assessment": _assessment("attack_feasibility"),
            "defense_assessment": _assessment("defense_robustness"),
        }
        fake_stage1 = _stage1_artifact()
        fake_result = _result([_round(1)])
        fake_stage2 = {"debate_result": fake_result}
        fake_identity = {"input_fingerprint": "c" * 64}
        with (
            patch.object(nodes, "_load_exact_stage1_artifact", return_value=fake_stage1),
            patch.object(
                nodes,
                "load_frozen_stage2_artifact",
                return_value=(fake_stage2, fake_identity, Path("stage2.json")),
            ),
        ):
            prepared = nodes.prepare_stage2_node(state)
        self.assertTrue(prepared["stage2_artifact_reused"])
        self.assertEqual(prepared["debate_rounds"], fake_result["rounds"])

    def test_active_graph_runs_stage2_only_after_stage1_join(self):
        graph = create_graph().get_graph()
        edges = {(edge.source, edge.target) for edge in graph.edges}
        self.assertIn(("attack_feasibility_critic", "stage1_complete"), edges)
        self.assertIn(("defense_robustness_critic", "stage1_complete"), edges)
        self.assertIn(("stage1_complete", "stage2_prepare"), edges)
        self.assertIn(("stage2_prepare", "stage2_debate"), edges)
        self.assertIn(("stage2_debate", "stage2_complete"), edges)
        self.assertIn(("stage2_complete", "stage3_build"), edges)
        self.assertIn(("stage3_build", "stage3_contextualize"), edges)
        self.assertIn(("stage3_contextualize", "stage4_context_prepare"), edges)

    def test_prepare_rejects_state_preassessment_that_differs_from_frozen_stage1(self):
        state = {
            "stage1_complete": True,
            "forecast_data": "FORECAST",
            "evidence_pack": _evidence_pack(),
            "attack_assessment": _assessment("attack_feasibility"),
            "defense_assessment": _assessment("defense_robustness"),
        }
        state["attack_assessment"]["unresolved_questions"] = ["mutated after Stage 1 freeze"]
        with patch.object(nodes, "_load_exact_stage1_artifact", return_value=_stage1_artifact()):
            with self.assertRaisesRegex(ValueError, "does not match the exact frozen Stage 1 artifact"):
                nodes.prepare_stage2_node(state)


class Stage2OrchestrationTests(unittest.IsolatedAsyncioTestCase):
    def _state(self) -> dict:
        return {
            "stage2_artifact_reused": False,
            "forecast_data": json.dumps(
                {
                    "case_id": "case__threat__pmt",
                    "evidence": [
                        {"evidence_id": "E1", "content": "first visible evidence"},
                        {"evidence_id": "E2", "content": "second visible evidence"},
                    ],
                }
            ),
            "evidence_pack": _evidence_pack(),
            "attack_assessment": _assessment("attack_feasibility"),
            "defense_assessment": _assessment("defense_robustness"),
        }

    def test_prompt_visibility_filter_hides_unsurfaced_evidence_content(self):
        forecast_data = json.dumps(
            {
                "case_id": "case__threat__pmt",
                "forecast_summary": {"gap_direction": "flat"},
                "evidence": [
                    {"evidence_id": "E1", "content": "visible-one"},
                    {"evidence_id": "E2", "content": "visible-two"},
                    {"evidence_id": "E3", "content": "MUST_NOT_BE_VISIBLE"},
                ],
            }
        )
        filtered = nodes._forecast_data_visible_to_stage2_step(
            forecast_data,
            allowed_evidence_ids=["E1", "E2"],
        )
        payload = json.loads(filtered)
        self.assertEqual([item["evidence_id"] for item in payload["evidence"]], ["E1", "E2"])
        self.assertNotIn("MUST_NOT_BE_VISIBLE", filtered)
        self.assertEqual(payload["stage2_evidence_visibility"]["mode"], "surfaced_only")

    async def _run_with_fake_llm(self, *, challenge_round1: bool):
        fake_llm = _FakeStage2LLM(challenge_round1=challenge_round1)
        with (
            patch.object(nodes, "_load_exact_stage1_artifact", return_value=_stage1_artifact()),
            patch.object(nodes, "get_llm", return_value=fake_llm),
            patch.object(nodes, "load_stage2_step_checkpoint", return_value=(None, Path("missing.json"))),
            patch.object(nodes, "freeze_stage2_step_checkpoint", return_value=Path("checkpoint.json")),
        ):
            result = await nodes.stage2_debate_node(self._state())
        return fake_llm, result

    async def test_one_round_flow_runs_turns_mediator_then_two_post_assessments(self):
        fake_llm, result = await self._run_with_fake_llm(challenge_round1=False)
        self.assertEqual(len(result["debate_rounds"]), 1)
        self.assertEqual(len(result["stage2_debate_result"]["exchanges"]), 4)
        self.assertEqual(
            [call["schema"] for call in fake_llm.calls],
            [
                "CriticDebateTurn",
                "CriticDebateTurn",
                "MediatorSummary",
                "CriticAssessment",
                "CriticAssessment",
            ],
        )
        for call in fake_llm.calls[:2]:
            system_prompt = call["messages"][0].content
            self.assertNotIn("Produce an **independent initial assessment**", system_prompt)
            self.assertNotIn("Return only the structured `CriticAssessment`", system_prompt)
            self.assertIn("Stage 2 structured debate task", system_prompt)

    async def test_mediator_sees_surfaced_evidence_content_but_not_unsurfaced_record(self):
        state = self._state()
        state["evidence_pack"]["evidence"].append({"evidence_id": "E3"})
        forecast_payload = json.loads(state["forecast_data"])
        forecast_payload["evidence"].append(
            {"evidence_id": "E3", "content": "UNSURFACED_SECRET_RECORD"}
        )
        state["forecast_data"] = json.dumps(forecast_payload)
        fake_llm = _FakeStage2LLM(challenge_round1=False)
        with (
            patch.object(nodes, "_load_exact_stage1_artifact", return_value=_stage1_artifact()),
            patch.object(nodes, "get_llm", return_value=fake_llm),
            patch.object(nodes, "load_stage2_step_checkpoint", return_value=(None, Path("missing.json"))),
            patch.object(nodes, "freeze_stage2_step_checkpoint", return_value=Path("checkpoint.json")),
        ):
            await nodes.stage2_debate_node(state)

        mediator_call = next(call for call in fake_llm.calls if call["schema"] == "MediatorSummary")
        mediator_prompt = mediator_call["messages"][0].content
        self.assertIn("first visible evidence", mediator_prompt)
        self.assertIn("second visible evidence", mediator_prompt)
        self.assertNotIn("UNSURFACED_SECRET_RECORD", mediator_prompt)
        self.assertNotIn('"evidence_id": "E3"', mediator_prompt)

    async def test_round1_mediator_continue_runs_exactly_one_second_round(self):
        fake_llm, result = await self._run_with_fake_llm(challenge_round1=True)
        self.assertEqual(len(result["debate_rounds"]), 2)
        self.assertEqual(len(result["stage2_debate_result"]["exchanges"]), 8)
        self.assertEqual(len(fake_llm.calls), 8)
        round2_turn_calls = [
            call
            for call in fake_llm.calls
            if call["schema"] == "CriticDebateTurn" and "Stage2-R2" in call["progress_label"]
        ]
        self.assertEqual(len(round2_turn_calls), 2)
        for call in round2_turn_calls:
            self.assertIn('"round_index": 1', call["messages"][0].content)
            self.assertIn('"next_round_focus"', call["messages"][0].content)
            self.assertIn("Reconsider C1 against the surfaced evidence", call["messages"][0].content)

    async def test_complete_step_checkpoints_resume_without_llm_server(self):
        payloads = {
            "round1_attack": _turn("attack_feasibility", 1),
            "round1_defense": _turn("defense_robustness", 1),
            "round1_mediator": _mediator(1),
            "post_attack": _assessment("attack_feasibility"),
            "post_defense": _assessment("defense_robustness"),
        }

        def checkpoint_loader(*, step_name, **kwargs):
            del kwargs
            return {"payload": payloads[step_name]}, Path(f"{step_name}.json")

        with (
            patch.object(nodes, "_load_exact_stage1_artifact", return_value=_stage1_artifact()),
            patch.object(nodes, "get_llm", side_effect=AssertionError("LLM server must not be touched")),
            patch.object(nodes, "load_stage2_step_checkpoint", side_effect=checkpoint_loader),
        ):
            result = await nodes.stage2_debate_node(self._state())

        self.assertEqual(len(result["debate_rounds"]), 1)
        self.assertEqual(result["stage2_debate_result"]["stance_changes"]["attack_feasibility"], 0)


if __name__ == "__main__":
    unittest.main()
