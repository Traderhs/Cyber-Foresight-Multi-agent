from __future__ import annotations

import copy
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from Stage1.schema import (
    CriticAssessment,
    CriticType,
    build_constrained_critic_response_schema,
    validate_critic_assessment,
)


STAGE2_SEMANTIC_VALIDATION_VERSION = "stage2-semantic-validation-v9"
STAGE2_MAX_ROUNDS = 2


class Stage2ValidationError(ValueError):
    """Raised when a Stage 2 structured output violates the fixed contract."""


class DebateResponseType(StrEnum):
    AGREE = "AGREE"
    CHALLENGE = "CHALLENGE"
    REVISE = "REVISE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class DisputeStatus(StrEnum):
    RESOLVED = "RESOLVED"
    UNRESOLVED = "UNRESOLVED"


class MediatorRoundAction(StrEnum):
    STOP = "STOP"
    CONTINUE = "CONTINUE"


class MediatorAdjudicationOutcome(StrEnum):
    ACCEPT_ATTACK = "ACCEPT_ATTACK"
    ACCEPT_DEFENSE = "ACCEPT_DEFENSE"
    ACCEPT_BOTH = "ACCEPT_BOTH"
    REJECT_ATTACK = "REJECT_ATTACK"
    REJECT_DEFENSE = "REJECT_DEFENSE"
    REJECT_BOTH = "REJECT_BOTH"
    REVISE_ATTACK = "REVISE_ATTACK"
    REVISE_DEFENSE = "REVISE_DEFENSE"
    REVISE_BOTH = "REVISE_BOTH"
    UNRESOLVED = "UNRESOLVED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class DebateResponseItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_claim_id: str = Field(min_length=1)
    response_type: DebateResponseType
    evidence_ids: list[str] = Field(default_factory=list)
    note: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_grounding(self) -> "DebateResponseItem":
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("evidence_ids must be unique within one debate response")
        if self.response_type == DebateResponseType.INSUFFICIENT_EVIDENCE and self.evidence_ids:
            raise ValueError("INSUFFICIENT_EVIDENCE responses must not cite evidence IDs")
        if self.response_type != DebateResponseType.INSUFFICIENT_EVIDENCE and not self.evidence_ids:
            raise ValueError(
                "AGREE/CHALLENGE/REVISE responses require at least one exact Stage 0 evidence ID"
            )
        return self


class CriticDebateTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    round_index: int = Field(ge=1, le=STAGE2_MAX_ROUNDS)
    responder_critic: CriticType
    target_critic: CriticType
    responses: list[DebateResponseItem] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_roles_and_targets(self) -> "CriticDebateTurn":
        if self.responder_critic == self.target_critic:
            raise ValueError("A critic may respond only to the other Stage 1 critic")
        target_ids = [item.target_claim_id for item in self.responses]
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("Each opponent claim must appear at most once in a debate turn")
        return self


class ClaimMatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    match_id: str = Field(min_length=1)
    attack_claim_ids: list[str] = Field(min_length=1)
    defense_claim_ids: list[str] = Field(min_length=1)
    status: DisputeStatus
    evidence_ids: list[str] = Field(default_factory=list)
    note: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_refs(self) -> "ClaimMatch":
        if len(self.attack_claim_ids) != len(set(self.attack_claim_ids)):
            raise ValueError("ClaimMatch attack_claim_ids must be unique")
        if len(self.defense_claim_ids) != len(set(self.defense_claim_ids)):
            raise ValueError("ClaimMatch defense_claim_ids must be unique")
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("ClaimMatch evidence_ids must be unique")
        return self


class EvidenceConflict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conflict_id: str = Field(min_length=1)
    attack_claim_ids: list[str] = Field(default_factory=list)
    defense_claim_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(min_length=1)
    note: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_refs(self) -> "EvidenceConflict":
        if not self.attack_claim_ids and not self.defense_claim_ids:
            raise ValueError("EvidenceConflict must reference at least one Stage 1 claim")
        if len(self.attack_claim_ids) != len(set(self.attack_claim_ids)):
            raise ValueError("EvidenceConflict attack_claim_ids must be unique")
        if len(self.defense_claim_ids) != len(set(self.defense_claim_ids)):
            raise ValueError("EvidenceConflict defense_claim_ids must be unique")
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("EvidenceConflict evidence_ids must be unique")
        return self


class EvidenceGap(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gap_id: str = Field(min_length=1)
    attack_claim_ids: list[str] = Field(default_factory=list)
    defense_claim_ids: list[str] = Field(default_factory=list)
    question: str = Field(min_length=1)

    @model_validator(mode="after")
    def require_related_claim(self) -> "EvidenceGap":
        if not self.attack_claim_ids and not self.defense_claim_ids:
            raise ValueError("EvidenceGap must reference at least one Stage 1 claim")
        if len(self.attack_claim_ids) != len(set(self.attack_claim_ids)):
            raise ValueError("EvidenceGap attack_claim_ids must be unique")
        if len(self.defense_claim_ids) != len(set(self.defense_claim_ids)):
            raise ValueError("EvidenceGap defense_claim_ids must be unique")
        return self


class MediatorAdjudication(BaseModel):
    """Evidence-bounded Mediator judgment over one dispute unit."""

    model_config = ConfigDict(extra="forbid")

    adjudication_id: str = Field(min_length=1)
    attack_claim_ids: list[str] = Field(default_factory=list)
    defense_claim_ids: list[str] = Field(default_factory=list)
    outcome: MediatorAdjudicationOutcome
    evidence_ids: list[str] = Field(default_factory=list)
    rationale: str = Field(min_length=1)
    required_revision: str | None = None

    @model_validator(mode="after")
    def validate_adjudication_shape(self) -> "MediatorAdjudication":
        if not self.attack_claim_ids and not self.defense_claim_ids:
            raise ValueError("MediatorAdjudication must reference at least one Stage 1 claim")
        if len(self.attack_claim_ids) != len(set(self.attack_claim_ids)):
            raise ValueError("MediatorAdjudication attack_claim_ids must be unique")
        if len(self.defense_claim_ids) != len(set(self.defense_claim_ids)):
            raise ValueError("MediatorAdjudication defense_claim_ids must be unique")
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("MediatorAdjudication evidence_ids must be unique")

        if self.outcome == MediatorAdjudicationOutcome.ACCEPT_ATTACK and not self.attack_claim_ids:
            raise ValueError("ACCEPT_ATTACK requires at least one Attack claim")
        if self.outcome == MediatorAdjudicationOutcome.ACCEPT_DEFENSE and not self.defense_claim_ids:
            raise ValueError("ACCEPT_DEFENSE requires at least one Defense claim")
        if self.outcome == MediatorAdjudicationOutcome.ACCEPT_BOTH and (
            not self.attack_claim_ids or not self.defense_claim_ids
        ):
            raise ValueError("ACCEPT_BOTH requires both Attack and Defense claims")
        if self.outcome == MediatorAdjudicationOutcome.REJECT_ATTACK and not self.attack_claim_ids:
            raise ValueError("REJECT_ATTACK requires at least one Attack claim")
        if self.outcome == MediatorAdjudicationOutcome.REJECT_DEFENSE and not self.defense_claim_ids:
            raise ValueError("REJECT_DEFENSE requires at least one Defense claim")
        if self.outcome == MediatorAdjudicationOutcome.REJECT_BOTH and (
            not self.attack_claim_ids or not self.defense_claim_ids
        ):
            raise ValueError("REJECT_BOTH requires both Attack and Defense claims")

        revision_outcomes = {
            MediatorAdjudicationOutcome.REVISE_ATTACK,
            MediatorAdjudicationOutcome.REVISE_DEFENSE,
            MediatorAdjudicationOutcome.REVISE_BOTH,
        }
        if self.outcome == MediatorAdjudicationOutcome.REVISE_ATTACK and not self.attack_claim_ids:
            raise ValueError("REVISE_ATTACK requires at least one Attack claim")
        if self.outcome == MediatorAdjudicationOutcome.REVISE_DEFENSE and not self.defense_claim_ids:
            raise ValueError("REVISE_DEFENSE requires at least one Defense claim")
        if self.outcome == MediatorAdjudicationOutcome.REVISE_BOTH and (
            not self.attack_claim_ids or not self.defense_claim_ids
        ):
            raise ValueError("REVISE_BOTH requires both Attack and Defense claims")
        if self.outcome in revision_outcomes:
            if not self.required_revision or not self.required_revision.strip():
                raise ValueError("REVISE_* adjudications require a concrete required_revision")
        elif self.required_revision is not None:
            raise ValueError("required_revision is allowed only for REVISE_* adjudications")

        if self.outcome == MediatorAdjudicationOutcome.INSUFFICIENT_EVIDENCE:
            if self.evidence_ids:
                raise ValueError("INSUFFICIENT_EVIDENCE adjudication must not cite evidence IDs")
        elif not self.evidence_ids:
            raise ValueError("Non-gap Mediator adjudications require at least one surfaced evidence ID")
        return self


class NextRoundFocus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    focus_id: str = Field(min_length=1)
    attack_claim_ids: list[str] = Field(default_factory=list)
    defense_claim_ids: list[str] = Field(default_factory=list)
    instruction: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_focus_refs(self) -> "NextRoundFocus":
        if not self.attack_claim_ids and not self.defense_claim_ids:
            raise ValueError("NextRoundFocus must reference at least one Stage 1 claim")
        if len(self.attack_claim_ids) != len(set(self.attack_claim_ids)):
            raise ValueError("NextRoundFocus attack_claim_ids must be unique")
        if len(self.defense_claim_ids) != len(set(self.defense_claim_ids)):
            raise ValueError("NextRoundFocus defense_claim_ids must be unique")
        return self


class MediatorSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    round_index: int = Field(ge=1, le=STAGE2_MAX_ROUNDS)
    claim_matches: list[ClaimMatch] = Field(default_factory=list)
    evidence_conflicts: list[EvidenceConflict] = Field(default_factory=list)
    evidence_gaps: list[EvidenceGap] = Field(default_factory=list)
    adjudications: list[MediatorAdjudication] = Field(min_length=1)
    round_action: MediatorRoundAction
    next_round_focus: list[NextRoundFocus] = Field(default_factory=list)
    overall_rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_record_ids(self) -> "MediatorSummary":
        for name, items in (
            ("match_id", self.claim_matches),
            ("conflict_id", self.evidence_conflicts),
            ("gap_id", self.evidence_gaps),
            ("adjudication_id", self.adjudications),
            ("focus_id", self.next_round_focus),
        ):
            values = [getattr(item, name) for item in items]
            if len(values) != len(set(values)):
                raise ValueError(f"Mediator {name} values must be unique")
        if self.round_action == MediatorRoundAction.CONTINUE and not self.next_round_focus:
            raise ValueError("CONTINUE requires at least one next_round_focus item")
        if self.round_action == MediatorRoundAction.STOP and self.next_round_focus:
            raise ValueError("STOP must not include next_round_focus items")
        return self


class DebateRound(BaseModel):
    model_config = ConfigDict(extra="forbid")

    round_index: int = Field(ge=1, le=STAGE2_MAX_ROUNDS)
    attack_turn: CriticDebateTurn
    defense_turn: CriticDebateTurn
    mediator_summary: MediatorSummary

    @model_validator(mode="after")
    def validate_round_alignment(self) -> "DebateRound":
        if self.attack_turn.round_index != self.round_index:
            raise ValueError("attack_turn round_index mismatch")
        if self.defense_turn.round_index != self.round_index:
            raise ValueError("defense_turn round_index mismatch")
        if self.mediator_summary.round_index != self.round_index:
            raise ValueError("mediator_summary round_index mismatch")
        if self.attack_turn.responder_critic != CriticType.ATTACK_FEASIBILITY:
            raise ValueError("attack_turn must be produced by attack_feasibility")
        if self.defense_turn.responder_critic != CriticType.DEFENSE_ROBUSTNESS:
            raise ValueError("defense_turn must be produced by defense_robustness")
        return self


class StanceChanges(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attack_feasibility: int = Field(ge=-2, le=2)
    defense_robustness: int = Field(ge=-2, le=2)


class DebateExchange(BaseModel):
    """Programmatic flattened view of every claim-level critic response."""

    model_config = ConfigDict(extra="forbid")

    round_index: int = Field(ge=1, le=STAGE2_MAX_ROUNDS)
    claim_id: str = Field(min_length=1)
    source_critic: CriticType
    target_critic: CriticType
    response_type: DebateResponseType
    evidence_ids: list[str] = Field(default_factory=list)
    note: str = Field(min_length=1)


class Stage2DebateResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    attack_pre_assessment: CriticAssessment
    defense_pre_assessment: CriticAssessment
    rounds: list[DebateRound] = Field(min_length=1, max_length=STAGE2_MAX_ROUNDS)
    exchanges: list[DebateExchange] = Field(min_length=1)
    resolved_claims: list[ClaimMatch] = Field(default_factory=list)
    unresolved_claims: list[ClaimMatch] = Field(default_factory=list)
    evidence_conflicts: list[EvidenceConflict] = Field(default_factory=list)
    evidence_gaps: list[EvidenceGap] = Field(default_factory=list)
    final_adjudications: list[MediatorAdjudication] = Field(min_length=1)
    attack_post_assessment: CriticAssessment
    defense_post_assessment: CriticAssessment
    stance_changes: StanceChanges


def _evidence_ids(evidence_pack: dict[str, Any]) -> list[str]:
    return sorted(
        {
            str(record.get("evidence_id"))
            for record in evidence_pack.get("evidence", [])
            if record.get("evidence_id")
        }
    )


def _claim_ids(assessment: CriticAssessment | dict[str, Any]) -> list[str]:
    parsed = assessment if isinstance(assessment, CriticAssessment) else CriticAssessment.model_validate(assessment)
    return sorted(claim.claim_id for claim in parsed.claims)


def surfaced_evidence_ids(
    *,
    attack_assessment: CriticAssessment | dict[str, Any],
    defense_assessment: CriticAssessment | dict[str, Any],
    attack_turn: CriticDebateTurn | dict[str, Any] | None = None,
    defense_turn: CriticDebateTurn | dict[str, Any] | None = None,
    previous_rounds: list[DebateRound | dict[str, Any]] | None = None,
) -> list[str]:
    surfaced = {
        evidence_id
        for assessment in (attack_assessment, defense_assessment)
        for claim in (
            assessment.claims
            if isinstance(assessment, CriticAssessment)
            else CriticAssessment.model_validate(assessment).claims
        )
        for evidence_id in claim.evidence_ids
    }
    turns: list[CriticDebateTurn] = []
    if attack_turn is not None:
        turns.append(
            attack_turn if isinstance(attack_turn, CriticDebateTurn) else CriticDebateTurn.model_validate(attack_turn)
        )
    if defense_turn is not None:
        turns.append(
            defense_turn if isinstance(defense_turn, CriticDebateTurn) else CriticDebateTurn.model_validate(defense_turn)
        )
    for round_item in previous_rounds or []:
        parsed_round = round_item if isinstance(round_item, DebateRound) else DebateRound.model_validate(round_item)
        turns.extend((parsed_round.attack_turn, parsed_round.defense_turn))
    surfaced.update(
        evidence_id
        for turn in turns
        for response in turn.responses
        for evidence_id in response.evidence_ids
    )
    return sorted(surfaced)


def build_constrained_post_assessment_schema(
    *,
    evidence_pack: dict[str, Any],
    expected_critic_type: CriticType,
    attack_pre_assessment: CriticAssessment | dict[str, Any],
    defense_pre_assessment: CriticAssessment | dict[str, Any],
    rounds: list[DebateRound | dict[str, Any]],
) -> dict[str, Any]:
    """Constrain post-assessment evidence to material surfaced before/during debate."""
    allowed_evidence = surfaced_evidence_ids(
        attack_assessment=attack_pre_assessment,
        defense_assessment=defense_pre_assessment,
        previous_rounds=rounds,
    )
    if not allowed_evidence:
        raise Stage2ValidationError("Stage 2 post-assessment has no surfaced evidence to cite")
    schema = build_constrained_critic_response_schema(
        evidence_pack=evidence_pack,
        expected_critic_type=expected_critic_type,
    )
    schema["$defs"]["ClaimAssessment"]["properties"]["evidence_ids"]["items"] = {
        "type": "string",
        "enum": allowed_evidence,
    }
    return schema


def validate_post_assessment(
    assessment: CriticAssessment | dict[str, Any],
    *,
    evidence_pack: dict[str, Any],
    expected_critic_type: CriticType,
    attack_pre_assessment: CriticAssessment | dict[str, Any],
    defense_pre_assessment: CriticAssessment | dict[str, Any],
    rounds: list[DebateRound | dict[str, Any]],
) -> CriticAssessment:
    parsed = validate_critic_assessment(
        assessment,
        evidence_pack=evidence_pack,
        expected_critic_type=expected_critic_type,
    )
    allowed_evidence = set(
        surfaced_evidence_ids(
            attack_assessment=attack_pre_assessment,
            defense_assessment=defense_pre_assessment,
            previous_rounds=rounds,
        )
    )
    post_evidence = {
        evidence_id for claim in parsed.claims for evidence_id in claim.evidence_ids
    }
    unsurfaced = sorted(post_evidence - allowed_evidence)
    if unsurfaced:
        raise Stage2ValidationError(
            "Post-assessment introduced Stage 0 evidence that was never surfaced in the frozen pre-assessments "
            f"or completed debate trace: {unsurfaced}"
        )
    return parsed


def build_constrained_debate_turn_schema(
    *,
    evidence_pack: dict[str, Any],
    opponent_assessment: CriticAssessment | dict[str, Any],
    expected_responder: CriticType,
    round_index: int,
    allowed_evidence_ids: list[str] | None = None,
) -> dict[str, Any]:
    case_id = str(evidence_pack.get("case_id") or "")
    all_evidence_ids = _evidence_ids(evidence_pack)
    evidence_ids = sorted(
        set(all_evidence_ids if allowed_evidence_ids is None else allowed_evidence_ids)
    )
    unknown_allowed = sorted(set(evidence_ids) - set(all_evidence_ids))
    if unknown_allowed:
        raise Stage2ValidationError(
            f"Constrained debate schema received unknown allowed evidence IDs: {unknown_allowed}"
        )
    opponent_claim_ids = _claim_ids(opponent_assessment)
    if not case_id or not evidence_ids or not opponent_claim_ids:
        raise Stage2ValidationError("Stage 2 debate turn requires a case, evidence IDs, and opponent claim IDs")
    target_critic = (
        CriticType.DEFENSE_ROBUSTNESS
        if expected_responder == CriticType.ATTACK_FEASIBILITY
        else CriticType.ATTACK_FEASIBILITY
    )
    schema = copy.deepcopy(CriticDebateTurn.model_json_schema())
    schema["properties"]["case_id"] = {"type": "string", "const": case_id}
    schema["properties"]["round_index"] = {"type": "integer", "const": round_index}
    schema["properties"]["responder_critic"] = {"type": "string", "const": expected_responder.value}
    schema["properties"]["target_critic"] = {"type": "string", "const": target_critic.value}
    response_props = schema["$defs"]["DebateResponseItem"]["properties"]
    response_props["target_claim_id"] = {"type": "string", "enum": opponent_claim_ids}
    response_props["evidence_ids"]["items"] = {"type": "string", "enum": evidence_ids}
    return schema


def validate_debate_turn(
    turn: CriticDebateTurn | dict[str, Any],
    *,
    evidence_pack: dict[str, Any],
    opponent_assessment: CriticAssessment | dict[str, Any],
    expected_responder: CriticType,
    round_index: int,
    allowed_evidence_ids: list[str] | None = None,
) -> CriticDebateTurn:
    try:
        parsed = turn if isinstance(turn, CriticDebateTurn) else CriticDebateTurn.model_validate(turn)
    except Exception as exc:
        raise Stage2ValidationError(f"Invalid CriticDebateTurn schema: {exc}") from exc
    expected_case_id = str(evidence_pack.get("case_id") or "")
    target_critic = (
        CriticType.DEFENSE_ROBUSTNESS
        if expected_responder == CriticType.ATTACK_FEASIBILITY
        else CriticType.ATTACK_FEASIBILITY
    )
    if parsed.case_id != expected_case_id:
        raise Stage2ValidationError("Debate turn case_id mismatch")
    if parsed.round_index != round_index:
        raise Stage2ValidationError("Debate turn round_index mismatch")
    if parsed.responder_critic != expected_responder or parsed.target_critic != target_critic:
        raise Stage2ValidationError("Debate turn critic-role mismatch")

    all_evidence = set(_evidence_ids(evidence_pack))
    allowed_evidence = set(allowed_evidence_ids) if allowed_evidence_ids is not None else all_evidence
    if not allowed_evidence.issubset(all_evidence):
        raise Stage2ValidationError("Debate turn allowed-evidence set contains unknown Stage 0 evidence IDs")
    expected_claims = set(_claim_ids(opponent_assessment))
    actual_claims = {item.target_claim_id for item in parsed.responses}
    if actual_claims != expected_claims:
        raise Stage2ValidationError(
            f"Debate turn must respond exactly once to every opponent claim; expected={sorted(expected_claims)}, "
            f"actual={sorted(actual_claims)}"
        )
    unknown_evidence = sorted(
        {
            evidence_id
            for item in parsed.responses
            for evidence_id in item.evidence_ids
            if evidence_id not in allowed_evidence
        }
    )
    if unknown_evidence:
        raise Stage2ValidationError(
            f"Debate turn cites evidence IDs outside the allowed evidence set for this round: {unknown_evidence}"
        )
    parsed_opponent = (
        opponent_assessment
        if isinstance(opponent_assessment, CriticAssessment)
        else CriticAssessment.model_validate(opponent_assessment)
    )
    opponent_evidence_by_claim = {
        claim.claim_id: set(claim.evidence_ids) for claim in parsed_opponent.claims
    }
    for response in parsed.responses:
        if response.response_type != DebateResponseType.AGREE:
            continue
        if not set(response.evidence_ids).intersection(opponent_evidence_by_claim[response.target_claim_id]):
            raise Stage2ValidationError(
                f"AGREE response for {response.target_claim_id!r} must re-cite at least one evidence ID used by "
                "the target Stage 1 claim; agreement cannot be grounded only in unrelated/additional evidence"
            )
    return parsed


def build_constrained_mediator_schema(
    *,
    evidence_pack: dict[str, Any],
    attack_assessment: CriticAssessment | dict[str, Any],
    defense_assessment: CriticAssessment | dict[str, Any],
    round_index: int,
    attack_turn: CriticDebateTurn | dict[str, Any] | None = None,
    defense_turn: CriticDebateTurn | dict[str, Any] | None = None,
    previous_rounds: list[DebateRound | dict[str, Any]] | None = None,
) -> dict[str, Any]:
    case_id = str(evidence_pack.get("case_id") or "")
    evidence_ids = _evidence_ids(evidence_pack)
    if attack_turn is not None and defense_turn is not None:
        evidence_ids = surfaced_evidence_ids(
            attack_assessment=attack_assessment,
            defense_assessment=defense_assessment,
            attack_turn=attack_turn,
            defense_turn=defense_turn,
            previous_rounds=previous_rounds,
        )
    attack_claim_ids = _claim_ids(attack_assessment)
    defense_claim_ids = _claim_ids(defense_assessment)
    schema = copy.deepcopy(MediatorSummary.model_json_schema())
    schema["properties"]["case_id"] = {"type": "string", "const": case_id}
    schema["properties"]["round_index"] = {"type": "integer", "const": round_index}
    for def_name in (
        "ClaimMatch",
        "EvidenceConflict",
        "EvidenceGap",
        "MediatorAdjudication",
        "NextRoundFocus",
    ):
        props = schema["$defs"][def_name]["properties"]
        props["attack_claim_ids"]["items"] = {"type": "string", "enum": attack_claim_ids}
        props["defense_claim_ids"]["items"] = {"type": "string", "enum": defense_claim_ids}
        if "evidence_ids" in props:
            props["evidence_ids"]["items"] = {"type": "string", "enum": evidence_ids}

        # These arrays may legitimately be empty for one-sided records, but the
        # fields themselves must always be emitted. Leaving them optional lets
        # constrained generation omit the claim references entirely and defer
        # the failure to the semantic repair turn.
        required = set(schema["$defs"][def_name].get("required", []))
        required.update({"attack_claim_ids", "defense_claim_ids"})
        schema["$defs"][def_name]["required"] = sorted(required)

    adjudication_def = schema["$defs"]["MediatorAdjudication"]
    adjudication_required = set(adjudication_def.get("required", []))
    adjudication_required.add("required_revision")
    adjudication_def["required"] = sorted(adjudication_required)
    adjudication_def["properties"]["required_revision"] = {
        "anyOf": [
            {"type": "string", "minLength": 1},
            {"type": "null"},
        ]
    }
    if round_index >= STAGE2_MAX_ROUNDS:
        schema["properties"]["round_action"] = {
            "type": "string",
            "const": MediatorRoundAction.STOP.value,
        }
    return schema


def validate_mediator_summary(
    summary: MediatorSummary | dict[str, Any],
    *,
    evidence_pack: dict[str, Any],
    attack_assessment: CriticAssessment | dict[str, Any],
    defense_assessment: CriticAssessment | dict[str, Any],
    attack_turn: CriticDebateTurn | dict[str, Any],
    defense_turn: CriticDebateTurn | dict[str, Any],
    round_index: int,
    previous_rounds: list[DebateRound | dict[str, Any]] | None = None,
) -> MediatorSummary:
    try:
        parsed = summary if isinstance(summary, MediatorSummary) else MediatorSummary.model_validate(summary)
    except Exception as exc:
        raise Stage2ValidationError(f"Invalid MediatorSummary schema: {exc}") from exc
    if parsed.case_id != str(evidence_pack.get("case_id") or "") or parsed.round_index != round_index:
        raise Stage2ValidationError("Mediator summary case/round mismatch")
    attack_ids = set(_claim_ids(attack_assessment))
    defense_ids = set(_claim_ids(defense_assessment))
    evidence_ids = set(_evidence_ids(evidence_pack))
    parsed_previous_rounds = [
        item if isinstance(item, DebateRound) else DebateRound.model_validate(item)
        for item in (previous_rounds or [])
    ]

    parsed_attack_turn = validate_debate_turn(
        attack_turn,
        evidence_pack=evidence_pack,
        opponent_assessment=defense_assessment,
        expected_responder=CriticType.ATTACK_FEASIBILITY,
        round_index=round_index,
        allowed_evidence_ids=(
            surfaced_evidence_ids(
                attack_assessment=attack_assessment,
                defense_assessment=defense_assessment,
                previous_rounds=parsed_previous_rounds,
            )
            if round_index > 1
            else None
        ),
    )
    parsed_defense_turn = validate_debate_turn(
        defense_turn,
        evidence_pack=evidence_pack,
        opponent_assessment=attack_assessment,
        expected_responder=CriticType.DEFENSE_ROBUSTNESS,
        round_index=round_index,
        allowed_evidence_ids=(
            surfaced_evidence_ids(
                attack_assessment=attack_assessment,
                defense_assessment=defense_assessment,
                previous_rounds=parsed_previous_rounds,
            )
            if round_index > 1
            else None
        ),
    )
    organizer_evidence_ids = set(
        surfaced_evidence_ids(
            attack_assessment=attack_assessment,
            defense_assessment=defense_assessment,
            attack_turn=parsed_attack_turn,
            defense_turn=parsed_defense_turn,
            previous_rounds=parsed_previous_rounds,
        )
    )

    parsed_attack_assessment = (
        attack_assessment
        if isinstance(attack_assessment, CriticAssessment)
        else CriticAssessment.model_validate(attack_assessment)
    )
    parsed_defense_assessment = (
        defense_assessment
        if isinstance(defense_assessment, CriticAssessment)
        else CriticAssessment.model_validate(defense_assessment)
    )
    attack_evidence_by_claim = {
        claim.claim_id: set(claim.evidence_ids) for claim in parsed_attack_assessment.claims
    }
    defense_evidence_by_claim = {
        claim.claim_id: set(claim.evidence_ids) for claim in parsed_defense_assessment.claims
    }
    all_turns: list[CriticDebateTurn] = [
        *(turn for round_item in parsed_previous_rounds for turn in (round_item.attack_turn, round_item.defense_turn)),
        parsed_attack_turn,
        parsed_defense_turn,
    ]
    for turn in all_turns:
        target_map = (
            defense_evidence_by_claim
            if turn.responder_critic == CriticType.ATTACK_FEASIBILITY
            else attack_evidence_by_claim
        )
        for response in turn.responses:
            target_map[response.target_claim_id].update(response.evidence_ids)

    for collection in (
        parsed.claim_matches,
        parsed.evidence_conflicts,
        parsed.evidence_gaps,
        parsed.adjudications,
        parsed.next_round_focus,
    ):
        for item in collection:
            if not set(item.attack_claim_ids).issubset(attack_ids):
                raise Stage2ValidationError("Mediator summary references unknown Attack claim IDs")
            if not set(item.defense_claim_ids).issubset(defense_ids):
                raise Stage2ValidationError("Mediator summary references unknown Defense claim IDs")
            item_evidence = getattr(item, "evidence_ids", [])
            if not set(item_evidence).issubset(evidence_ids):
                raise Stage2ValidationError("Mediator summary references unknown Stage 0 evidence IDs")
            if not set(item_evidence).issubset(organizer_evidence_ids):
                raise Stage2ValidationError(
                    "Mediator may judge only from evidence already surfaced in Stage 1 claims, previous critic turns, "
                    "or the current critic turns"
                )
            local_evidence = set().union(
                *(attack_evidence_by_claim[claim_id] for claim_id in item.attack_claim_ids),
                *(defense_evidence_by_claim[claim_id] for claim_id in item.defense_claim_ids),
            )
            if not set(item_evidence).issubset(local_evidence):
                raise Stage2ValidationError(
                    "Mediator attached evidence that was not surfaced for the claim IDs referenced by this item"
                )

            if (
                isinstance(item, MediatorAdjudication)
                and item.outcome != MediatorAdjudicationOutcome.INSUFFICIENT_EVIDENCE
            ):
                cited = set(item.evidence_ids)
                uncovered_attack = [
                    claim_id
                    for claim_id in item.attack_claim_ids
                    if not cited.intersection(attack_evidence_by_claim[claim_id])
                ]
                uncovered_defense = [
                    claim_id
                    for claim_id in item.defense_claim_ids
                    if not cited.intersection(defense_evidence_by_claim[claim_id])
                ]
                if uncovered_attack or uncovered_defense:
                    raise Stage2ValidationError(
                        f"Mediator adjudication {item.adjudication_id!r} must cite claim-local surfaced evidence for "
                        "every referenced claim; uncovered Attack claims="
                        f"{uncovered_attack}, Defense claims={uncovered_defense}"
                    )

    # The Mediator is an evidence-bounded adjudicator. Python validates that it
    # considered every substantive critic objection/gap, but it does not replace
    # the Mediator's judgment with a hard-coded winner/status rule.
    adjudication_units = [
        (tuple(sorted(item.attack_claim_ids)), tuple(sorted(item.defense_claim_ids)))
        for item in parsed.adjudications
    ]
    if len(adjudication_units) != len(set(adjudication_units)):
        raise Stage2ValidationError(
            "Mediator may not emit multiple adjudications for the exact same Attack/Defense claim set in one round"
        )

    claim_match_units = [
        (tuple(sorted(item.attack_claim_ids)), tuple(sorted(item.defense_claim_ids)))
        for item in parsed.claim_matches
    ]
    if len(claim_match_units) != len(set(claim_match_units)):
        raise Stage2ValidationError(
            "Mediator may not emit duplicate ClaimMatch records for the exact same Attack/Defense claim set"
        )

    adjudicated_attack_ids = {
        claim_id for item in parsed.adjudications for claim_id in item.attack_claim_ids
    }
    adjudicated_defense_ids = {
        claim_id for item in parsed.adjudications for claim_id in item.defense_claim_ids
    }
    needs_adjudication_defense_ids = {
        item.target_claim_id
        for item in parsed_attack_turn.responses
        if item.response_type
        in {
            DebateResponseType.CHALLENGE,
            DebateResponseType.REVISE,
            DebateResponseType.INSUFFICIENT_EVIDENCE,
        }
    }
    needs_adjudication_attack_ids = {
        item.target_claim_id
        for item in parsed_defense_turn.responses
        if item.response_type
        in {
            DebateResponseType.CHALLENGE,
            DebateResponseType.REVISE,
            DebateResponseType.INSUFFICIENT_EVIDENCE,
        }
    }
    if not needs_adjudication_defense_ids.issubset(adjudicated_defense_ids):
        raise Stage2ValidationError(
            "Mediator omitted a challenged/revised/insufficient Defense claim from adjudications"
        )
    if not needs_adjudication_attack_ids.issubset(adjudicated_attack_ids):
        raise Stage2ValidationError(
            "Mediator omitted a challenged/revised/insufficient Attack claim from adjudications"
        )

    unresolved_outcomes = {
        MediatorAdjudicationOutcome.REVISE_ATTACK,
        MediatorAdjudicationOutcome.REVISE_DEFENSE,
        MediatorAdjudicationOutcome.REVISE_BOTH,
        MediatorAdjudicationOutcome.UNRESOLVED,
        MediatorAdjudicationOutcome.INSUFFICIENT_EVIDENCE,
    }
    for match in parsed.claim_matches:
        covering_adjudications = [
            item
            for item in parsed.adjudications
            if (
                set(match.attack_claim_ids) == set(item.attack_claim_ids)
                and set(match.defense_claim_ids) == set(item.defense_claim_ids)
            )
        ]
        if len(covering_adjudications) != 1:
            raise Stage2ValidationError(
                f"Mediator ClaimMatch {match.match_id!r} must map to exactly one adjudication "
                "with the same Attack/Defense claim set"
            )
        adjudication = covering_adjudications[0]
        expected_status = (
            DisputeStatus.UNRESOLVED
            if adjudication.outcome in unresolved_outcomes
            else DisputeStatus.RESOLVED
        )
        if match.status != expected_status:
            raise Stage2ValidationError(
                f"Mediator ClaimMatch {match.match_id!r} status conflicts with its adjudication outcomes: "
                f"expected {expected_status.value}, got {match.status.value}"
            )

    for adjudication in parsed.adjudications:
        if adjudication.outcome != MediatorAdjudicationOutcome.INSUFFICIENT_EVIDENCE:
            continue
        covering_gaps = [
            gap
            for gap in parsed.evidence_gaps
            if (
                set(adjudication.attack_claim_ids) == set(gap.attack_claim_ids)
                and set(adjudication.defense_claim_ids) == set(gap.defense_claim_ids)
            )
        ]
        if not covering_gaps:
            raise Stage2ValidationError(
                f"Mediator INSUFFICIENT_EVIDENCE adjudication {adjudication.adjudication_id!r} "
                "must be represented by an evidence gap with the same referenced claim set"
            )

    if round_index >= STAGE2_MAX_ROUNDS and parsed.round_action != MediatorRoundAction.STOP:
        raise Stage2ValidationError("The final allowed debate round must STOP")

    revision_or_unresolved = {
        MediatorAdjudicationOutcome.REVISE_ATTACK,
        MediatorAdjudicationOutcome.REVISE_DEFENSE,
        MediatorAdjudicationOutcome.REVISE_BOTH,
        MediatorAdjudicationOutcome.UNRESOLVED,
    }
    if parsed.round_action == MediatorRoundAction.CONTINUE:
        if not any(item.outcome in revision_or_unresolved for item in parsed.adjudications):
            raise Stage2ValidationError(
                "Mediator CONTINUE requires at least one REVISE_* or UNRESOLVED adjudication"
            )
        for focus in parsed.next_round_focus:
            covering_adjudications = [
                item
                for item in parsed.adjudications
                if item.outcome in revision_or_unresolved
                and set(focus.attack_claim_ids).issubset(item.attack_claim_ids)
                and set(focus.defense_claim_ids).issubset(item.defense_claim_ids)
            ]
            if not covering_adjudications:
                raise Stage2ValidationError(
                    "next_round_focus must stay wholly within one REVISE_* or UNRESOLVED adjudication"
                )
    return parsed


def round_requires_followup(round_result: DebateRound | dict[str, Any]) -> bool:
    parsed = round_result if isinstance(round_result, DebateRound) else DebateRound.model_validate(round_result)
    return parsed.mediator_summary.round_action == MediatorRoundAction.CONTINUE


def flatten_debate_exchanges(rounds: list[DebateRound]) -> list[DebateExchange]:
    exchanges: list[DebateExchange] = []
    for round_result in rounds:
        for turn in (round_result.attack_turn, round_result.defense_turn):
            for response in turn.responses:
                exchanges.append(
                    DebateExchange(
                        round_index=round_result.round_index,
                        claim_id=response.target_claim_id,
                        source_critic=turn.responder_critic,
                        target_critic=turn.target_critic,
                        response_type=response.response_type,
                        evidence_ids=response.evidence_ids,
                        note=response.note,
                    )
                )
    return exchanges


def validate_stage2_result(
    result: Stage2DebateResult | dict[str, Any],
    *,
    evidence_pack: dict[str, Any],
    attack_pre_assessment: CriticAssessment | dict[str, Any],
    defense_pre_assessment: CriticAssessment | dict[str, Any],
) -> Stage2DebateResult:
    try:
        parsed = result if isinstance(result, Stage2DebateResult) else Stage2DebateResult.model_validate(result)
    except Exception as exc:
        raise Stage2ValidationError(f"Invalid Stage2DebateResult schema: {exc}") from exc
    expected_case_id = str(evidence_pack.get("case_id") or "")
    if parsed.case_id != expected_case_id:
        raise Stage2ValidationError("Stage 2 result case_id mismatch")

    attack_pre = validate_critic_assessment(
        attack_pre_assessment,
        evidence_pack=evidence_pack,
        expected_critic_type=CriticType.ATTACK_FEASIBILITY,
    )
    defense_pre = validate_critic_assessment(
        defense_pre_assessment,
        evidence_pack=evidence_pack,
        expected_critic_type=CriticType.DEFENSE_ROBUSTNESS,
    )
    if parsed.attack_pre_assessment.model_dump(mode="json") != attack_pre.model_dump(mode="json"):
        raise Stage2ValidationError("Stage 2 Attack pre-assessment was modified")
    if parsed.defense_pre_assessment.model_dump(mode="json") != defense_pre.model_dump(mode="json"):
        raise Stage2ValidationError("Stage 2 Defense pre-assessment was modified")

    for expected_round_index, round_result in enumerate(parsed.rounds, start=1):
        if round_result.round_index != expected_round_index:
            raise Stage2ValidationError("Stage 2 rounds must be contiguous starting at round 1")
        round_allowed_evidence = (
            surfaced_evidence_ids(
                attack_assessment=attack_pre,
                defense_assessment=defense_pre,
                previous_rounds=list(parsed.rounds[: expected_round_index - 1]),
            )
            if expected_round_index > 1
            else None
        )
        validate_debate_turn(
            round_result.attack_turn,
            evidence_pack=evidence_pack,
            opponent_assessment=defense_pre,
            expected_responder=CriticType.ATTACK_FEASIBILITY,
            round_index=expected_round_index,
            allowed_evidence_ids=round_allowed_evidence,
        )
        validate_debate_turn(
            round_result.defense_turn,
            evidence_pack=evidence_pack,
            opponent_assessment=attack_pre,
            expected_responder=CriticType.DEFENSE_ROBUSTNESS,
            round_index=expected_round_index,
            allowed_evidence_ids=round_allowed_evidence,
        )
        validate_mediator_summary(
            round_result.mediator_summary,
            evidence_pack=evidence_pack,
            attack_assessment=attack_pre,
            defense_assessment=defense_pre,
            attack_turn=round_result.attack_turn,
            defense_turn=round_result.defense_turn,
            round_index=expected_round_index,
            previous_rounds=list(parsed.rounds[: expected_round_index - 1]),
        )

    expected_exchanges = [item.model_dump(mode="json") for item in flatten_debate_exchanges(parsed.rounds)]
    actual_exchanges = [item.model_dump(mode="json") for item in parsed.exchanges]
    if actual_exchanges != expected_exchanges:
        raise Stage2ValidationError("exchanges must be the exact programmatic flattening of all debate turns")

    if len(parsed.rounds) == 1 and round_requires_followup(parsed.rounds[0]):
        raise Stage2ValidationError("Mediator requested CONTINUE after round 1, so the fixed second round is required")
    if len(parsed.rounds) == 2 and not round_requires_followup(parsed.rounds[0]):
        raise Stage2ValidationError("Round 2 is allowed only when the round-1 Mediator requests CONTINUE")

    final_mediator = parsed.rounds[-1].mediator_summary
    expected_resolved = [
        item.model_dump(mode="json")
        for item in final_mediator.claim_matches
        if item.status == DisputeStatus.RESOLVED
    ]
    expected_unresolved = [
        item.model_dump(mode="json")
        for item in final_mediator.claim_matches
        if item.status == DisputeStatus.UNRESOLVED
    ]
    if [item.model_dump(mode="json") for item in parsed.resolved_claims] != expected_resolved:
        raise Stage2ValidationError("resolved_claims must equal the final Mediator RESOLVED claim matches")
    if [item.model_dump(mode="json") for item in parsed.unresolved_claims] != expected_unresolved:
        raise Stage2ValidationError("unresolved_claims must equal the final Mediator UNRESOLVED claim matches")
    if [item.model_dump(mode="json") for item in parsed.evidence_conflicts] != [
        item.model_dump(mode="json") for item in final_mediator.evidence_conflicts
    ]:
        raise Stage2ValidationError("evidence_conflicts must equal the final Mediator evidence conflicts")
    if [item.model_dump(mode="json") for item in parsed.evidence_gaps] != [
        item.model_dump(mode="json") for item in final_mediator.evidence_gaps
    ]:
        raise Stage2ValidationError("evidence_gaps must equal the final Mediator evidence gaps")
    if [item.model_dump(mode="json") for item in parsed.final_adjudications] != [
        item.model_dump(mode="json") for item in final_mediator.adjudications
    ]:
        raise Stage2ValidationError(
            "final_adjudications must equal the complete final Mediator adjudication snapshot"
        )

    attack_post = validate_post_assessment(
        parsed.attack_post_assessment,
        evidence_pack=evidence_pack,
        expected_critic_type=CriticType.ATTACK_FEASIBILITY,
        attack_pre_assessment=attack_pre,
        defense_pre_assessment=defense_pre,
        rounds=list(parsed.rounds),
    )
    defense_post = validate_post_assessment(
        parsed.defense_post_assessment,
        evidence_pack=evidence_pack,
        expected_critic_type=CriticType.DEFENSE_ROBUSTNESS,
        attack_pre_assessment=attack_pre,
        defense_pre_assessment=defense_pre,
        rounds=list(parsed.rounds),
    )
    expected_changes = {
        "attack_feasibility": attack_post.stance - attack_pre.stance,
        "defense_robustness": defense_post.stance - defense_pre.stance,
    }
    if parsed.stance_changes.model_dump() != expected_changes:
        raise Stage2ValidationError(
            f"stance_changes mismatch: expected={expected_changes}, actual={parsed.stance_changes.model_dump()}"
        )
    return parsed
