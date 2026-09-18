from __future__ import annotations

from typing import Any

from Stage3.context_schema import ContextualDecisionObject, Stage3ContextualDecisionBundle
from Stage4.schema import DecisionLensEvaluation, LensType, Stage4EvaluationBundle
from Stage5.schema import (
    ScenarioLensDiagnostic,
    Stage5DiagnosticBundle,
    SystemEvidenceDiagnostic,
)
from Stage6.schema import (
    DecisionRecommendation,
    STAGE6_D_LENS_INTERPRETATION,
    STAGE6_DECISION_POLICY_VERSION,
    STAGE6_LENS_SET_ROLE,
    STAGE6_RECOMMENDATION_INTERPRETATION,
    STAGE6_SYNTHESIS_AUTHORITY,
    STAGE6_SYNTHESIS_CONTRACT_VERSION,
    DecisionSynthesis,
    Stage6SynthesisBundle,
    Stage6ValidationError,
    SynthesisNarrative,
    extract_normalized_numeric_tokens,
    extract_report_evidence_ids,
    validate_synthesis_narrative,
)


_LENS_FIELD = {
    LensType.TECHNICAL_FEASIBILITY: "technical_feasibility",
    LensType.INSTITUTIONAL_REGIONAL: "institutional_regional",
    LensType.FINANCIAL_ADOPTION: "financial_adoption",
}

_NON_CLAIMS = [
    "No causal optimality claim.",
    "No majority-vote decision rule.",
    "The three Stage 4 lenses are predeclared functional decision dimensions, not a statistical sample of agents or stakeholders.",
    "D_lens is descriptive within-set dispersion, not a population uncertainty estimate, calibrated probability, or predictive uncertainty.",
    "The recommendation is conditional on the frozen Stage 6 decision policy and does not establish universal or stakeholder-optimal action validity.",
    "No practitioner validation claim.",
    "No guaranteed deployment success or utility improvement claim.",
    "No exact ROI, budget, staffing, or risk estimate unless it already exists in frozen upstream evidence.",
]


def _evidence_record_field(record: Any, key: str) -> Any:
    if isinstance(record, dict):
        return record.get(key)
    return getattr(record, key)


def _decision_recommendation(
    *,
    evaluation: DecisionLensEvaluation,
) -> tuple[DecisionRecommendation, str]:
    """Apply the predeclared asymmetric Stage 6 decision policy.

    Technical feasibility and Institutional/Regional feasibility are critical
    gates: a challenge on either means the frozen candidate action is not
    recommended in that scenario; an unresolved/neutral critical gate means
    the action is held. Once both critical gates support the action, the
    Financial/Adoption lens determines whether to recommend the action,
    recommend only a pilot, or hold it because adoption burden is challenged.

    Evidence sufficiency is preserved as a qualifier rather than a second vote.
    Stage 4 already forces genuinely insufficient evidence/context to stance=0.
    """

    technical = evaluation.technical_feasibility.stance
    institutional = evaluation.institutional_regional.stance
    financial = evaluation.financial_adoption.stance

    if technical == -1 or institutional == -1:
        return DecisionRecommendation.DO_NOT_RECOMMEND, "CRITICAL_GATE_CHALLENGED"
    if technical == 0 or institutional == 0:
        return DecisionRecommendation.HOLD, "CRITICAL_GATE_UNRESOLVED"
    if technical != 1 or institutional != 1:
        raise Stage6ValidationError("Unexpected critical-gate stance combination")

    if financial == 1:
        return DecisionRecommendation.RECOMMEND, "ALL_GATES_SUPPORTED"
    if financial == 0:
        return DecisionRecommendation.RECOMMEND_PILOT, "ADOPTION_GATE_UNRESOLVED"
    if financial == -1:
        return DecisionRecommendation.HOLD, "ADOPTION_GATE_CHALLENGED"
    raise Stage6ValidationError("Unexpected Financial/Adoption stance")


def _stance_state(value: int) -> str:
    if value == 1:
        return "SUPPORTED"
    if value == 0:
        return "UNRESOLVED"
    if value == -1:
        return "CHALLENGED"
    raise Stage6ValidationError(f"Unexpected Stage 4 stance: {value}")


def _decision_basis_contract(
    *,
    evaluation: DecisionLensEvaluation,
    decision_rule_id: str,
) -> dict[str, Any]:
    """Expose the exact deterministic policy basis without encoding case prose.

    The report may discuss every lens, gap, and condition, but only the lens
    states listed under policy_trigger_lenses are allowed to be described as
    the direct reason the frozen decision rule fired.  Other states remain
    context, prerequisites, or scope qualifiers.
    """

    stances = {
        LensType.TECHNICAL_FEASIBILITY: evaluation.technical_feasibility.stance,
        LensType.INSTITUTIONAL_REGIONAL: evaluation.institutional_regional.stance,
        LensType.FINANCIAL_ADOPTION: evaluation.financial_adoption.stance,
    }

    def row(lens: LensType) -> dict[str, Any]:
        stance = int(stances[lens])
        return {
            "lens": lens.value,
            "stance": stance,
            "state": _stance_state(stance),
        }

    critical = (LensType.TECHNICAL_FEASIBILITY, LensType.INSTITUTIONAL_REGIONAL)
    financial = LensType.FINANCIAL_ADOPTION

    if decision_rule_id == "CRITICAL_GATE_CHALLENGED":
        trigger = [row(lens) for lens in critical if stances[lens] == -1]
        prerequisites: list[dict[str, Any]] = []
        trigger_logic = "ANY_LISTED_TRIGGER_IS_POLICY_SUFFICIENT"
    elif decision_rule_id == "CRITICAL_GATE_UNRESOLVED":
        trigger = [row(lens) for lens in critical if stances[lens] == 0]
        prerequisites = []
        trigger_logic = "ANY_LISTED_TRIGGER_IS_POLICY_SUFFICIENT"
    elif decision_rule_id == "ALL_GATES_SUPPORTED":
        trigger = [row(financial)]
        prerequisites = [row(lens) for lens in critical]
        trigger_logic = "TRIGGER_AND_ALL_PREREQUISITES_REQUIRED"
    elif decision_rule_id == "ADOPTION_GATE_UNRESOLVED":
        trigger = [row(financial)]
        prerequisites = [row(lens) for lens in critical]
        trigger_logic = "TRIGGER_AND_ALL_PREREQUISITES_REQUIRED"
    elif decision_rule_id == "ADOPTION_GATE_CHALLENGED":
        trigger = [row(financial)]
        prerequisites = [row(lens) for lens in critical]
        trigger_logic = "TRIGGER_AND_ALL_PREREQUISITES_REQUIRED"
    else:
        raise Stage6ValidationError(f"Unknown Stage 6 decision_rule_id: {decision_rule_id}")

    trigger_lenses = {item["lens"] for item in trigger}
    prerequisite_lenses = {item["lens"] for item in prerequisites}
    non_triggering = [
        row(lens)
        for lens in LensType
        if lens.value not in trigger_lenses and lens.value not in prerequisite_lenses
    ]
    return {
        "decision_rule_id": decision_rule_id,
        "policy_trigger_lenses": trigger,
        "policy_prerequisite_lenses": prerequisites,
        "trigger_logic": trigger_logic,
        "non_triggering_lens_states": non_triggering,
        "evidence_gaps_role": "SCOPE_QUALIFIER_NOT_POLICY_TRIGGER",
        "upstream_conditions_role": "REASSESSMENT_CONTEXT_NOT_POLICY_TRIGGER",
        "stance_semantics": {
            "-1": "CHALLENGED",
            "0": "UNRESOLVED",
            "+1": "SUPPORTED",
        },
    }


def _lens_assessment(evaluation: DecisionLensEvaluation, lens: LensType):
    return getattr(evaluation, _LENS_FIELD[lens])


def _gaps(obj: ContextualDecisionObject, evaluation: DecisionLensEvaluation) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    def add(source: str, text: str, raw_id: str | None = None) -> None:
        text = str(text).strip()
        if not text:
            return
        gap_id = raw_id or f"G{len(rows) + 1:02d}"
        if any(item["gap_id"] == gap_id for item in rows):
            gap_id = f"G{len(rows) + 1:02d}"
        rows.append({"gap_id": gap_id, "source": source, "text": text})

    for gap in obj.unresolved_evidence_gaps:
        add("stage2_mediator", gap.question, gap.gap_id)
    for question in obj.stage2_critique_summary.attack_post_unresolved_questions:
        add("stage2_attack_post", question)
    for question in obj.stage2_critique_summary.defense_post_unresolved_questions:
        add("stage2_defense_post", question)
    for lens in LensType:
        assessment = _lens_assessment(evaluation, lens)
        for text in assessment.evidence_gaps:
            add(lens.value, text)
    return rows


def _conditions(evaluation: DecisionLensEvaluation) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for lens in LensType:
        assessment = _lens_assessment(evaluation, lens)
        for text in assessment.conditional_requirements:
            key = (lens.value, text)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "condition_id": f"C{len(rows) + 1:02d}",
                    "source_lens": lens.value,
                    "text": text,
                }
            )
    return rows


def build_stage6_synthesis_frame(
    *,
    decision_object: ContextualDecisionObject | dict[str, Any],
    evaluation: DecisionLensEvaluation | dict[str, Any],
    scenario_diagnostic: ScenarioLensDiagnostic | dict[str, Any],
    system_evidence_diagnostic: SystemEvidenceDiagnostic | dict[str, Any],
) -> dict[str, Any]:
    try:
        obj = (
            decision_object
            if isinstance(decision_object, ContextualDecisionObject)
            else ContextualDecisionObject.model_validate(decision_object)
        )
        ev = (
            evaluation
            if isinstance(evaluation, DecisionLensEvaluation)
            else DecisionLensEvaluation.model_validate(evaluation)
        )
        diag = (
            scenario_diagnostic
            if isinstance(scenario_diagnostic, ScenarioLensDiagnostic)
            else ScenarioLensDiagnostic.model_validate(scenario_diagnostic)
        )
        system = (
            system_evidence_diagnostic
            if isinstance(system_evidence_diagnostic, SystemEvidenceDiagnostic)
            else SystemEvidenceDiagnostic.model_validate(system_evidence_diagnostic)
        )
    except Exception as exc:
        raise Stage6ValidationError(f"Invalid Stage 6 frame input: {exc}") from exc
    if len({obj.decision_object_id, ev.decision_object_id, diag.decision_object_id}) != 1:
        raise Stage6ValidationError("Stage 6 decision_object_id alignment failure")

    supporting = sorted(
        [lens for lens in LensType if _lens_assessment(ev, lens).stance == 1],
        key=lambda item: item.value,
    )
    dissenting = sorted(
        [lens for lens in LensType if _lens_assessment(ev, lens).stance == -1],
        key=lambda item: item.value,
    )
    indeterminate = sorted(
        [lens for lens in LensType if _lens_assessment(ev, lens).stance == 0],
        key=lambda item: item.value,
    )
    gaps = _gaps(obj, ev)
    conditions = _conditions(ev)
    recommendation, decision_rule_id = _decision_recommendation(
        evaluation=ev,
    )
    decision_basis_contract = _decision_basis_contract(
        evaluation=ev,
        decision_rule_id=decision_rule_id,
    )
    deterministic_output = {
        "decision_object_id": obj.decision_object_id,
        "scenario_id": obj.scenario_id,
        "jurisdiction_code": obj.context_scenario.jurisdiction_code,
        "candidate_action_id": obj.candidate_action.action_id,
        "candidate_action_name": obj.candidate_action.action_name,
        "forecast_summary": obj.forecast_interpretation,
        "predictive_uncertainty_summary": obj.predictive_uncertainty,
        "critique_summary": {
            "attack_post_stance": obj.stage2_critique_summary.attack_post_stance,
            "defense_post_stance": obj.stage2_critique_summary.defense_post_stance,
            "final_mediator_adjudications": [
                item.model_dump(mode="json")
                for item in obj.stage2_critique_summary.final_mediator_adjudications
            ],
            "matched_resolved_claim_ids": obj.stage2_critique_summary.matched_resolved_claim_ids,
            "matched_unresolved_claim_ids": obj.stage2_critique_summary.matched_unresolved_claim_ids,
            "required_revisions": [
                item.model_dump(mode="json") for item in obj.stage2_critique_summary.required_revisions
            ],
            "attack_post_unresolved_questions": obj.stage2_critique_summary.attack_post_unresolved_questions,
            "defense_post_unresolved_questions": obj.stage2_critique_summary.defense_post_unresolved_questions,
        },
        "supporting_lenses": [lens.value for lens in supporting],
        "dissenting_lenses": [lens.value for lens in dissenting],
        "indeterminate_lenses": [lens.value for lens in indeterminate],
        "decision_recommendation": recommendation.value,
        "decision_policy_version": STAGE6_DECISION_POLICY_VERSION,
        "decision_rule_id": decision_rule_id,
        "recommendation_scope": "STRATEGIC_CANDIDATE_ACTION",
        "recommendation_interpretation": STAGE6_RECOMMENDATION_INTERPRETATION,
        "lens_set_role": STAGE6_LENS_SET_ROLE,
        "d_lens_interpretation": STAGE6_D_LENS_INTERPRETATION,
        "d_lens": diag.d_lens,
        "evidence_quality_summary": {
            "evidence_sufficiency_by_lens": {
                key: value.value for key, value in diag.evidence_sufficiency_by_lens.items()
            },
            "d_lens_evaluable": diag.d_lens_evaluable,
            "joint_abstention": diag.joint_abstention,
            "agreement_class": diag.agreement_class.value,
            "stage0_evidence_slot_coverage_ratio": system.evidence_slot_coverage_ratio,
            "source_family_count": system.source_family_count,
            "claim_evidence_reference_pass_rate": system.claim_evidence_reference_pass_rate,
            "semantic_grounding_audit_status": system.claim_evidence_audit_status.value,
        },
        "lens_evidence_trace": {
            lens.value: _lens_assessment(ev, lens).model_dump(mode="json")
            for lens in LensType
        },
        "unresolved_evidence_gaps": gaps,
        "upstream_conditions": conditions,
        "synthesis_authority": STAGE6_SYNTHESIS_AUTHORITY,
        "non_claims": list(_NON_CLAIMS),
        "provenance": {
            "scenario_id": obj.scenario_id,
            "jurisdiction_code": obj.context_scenario.jurisdiction_code,
            "candidate_selection_rule": obj.candidate_selection_rule,
            "evaluation_evidence_ids": sorted(obj.evaluation_evidence_ids),
            "cited_evidence_ids_by_lens": diag.cited_evidence_ids_by_lens,
        },
    }
    return {
        "deterministic_output": deterministic_output,
        "narrative_source": {
            "frozen_action": obj.candidate_action.model_dump(mode="json"),
            "deployment_context": obj.deployment_context.model_dump(mode="json"),
            "known_constraints": obj.known_constraints,
            "stage5_scenario_diagnostic": diag.model_dump(mode="json"),
            "decision_basis_contract": decision_basis_contract,
        },
        # Validator-only provenance.  This is deliberately excluded from the
        # LLM prompt: it allows Stage 6 to verify numeric claims against the
        # exact cited frozen evidence without giving the model an extra numeric
        # menu that could create anchoring or repetition pressure.
        "validation_context": {
            "evidence_numeric_tokens_by_id": {
                str(_evidence_record_field(record, "evidence_id")): sorted(
                    extract_normalized_numeric_tokens(
                        str(_evidence_record_field(record, "content") or "")
                    )
                )
                for record in obj.evaluation_evidence_records
            }
        },
    }


def build_decision_synthesis(
    *,
    frame: dict[str, Any],
    narrative: SynthesisNarrative | dict[str, Any],
) -> DecisionSynthesis:
    parsed = validate_synthesis_narrative(narrative, frame=frame)
    value = dict(frame["deterministic_output"])
    value["strategic_intelligence_report"] = parsed.strategic_intelligence_report
    value["report_cited_evidence_ids"] = extract_report_evidence_ids(
        report=parsed.strategic_intelligence_report,
        frame=frame,
    )
    return DecisionSynthesis.model_validate(value)


def build_stage6_frames(
    *,
    stage3_context_bundle: Stage3ContextualDecisionBundle | dict[str, Any],
    stage4_context_bundle: Stage4EvaluationBundle | dict[str, Any],
    stage5_diagnostic_bundle: Stage5DiagnosticBundle | dict[str, Any],
) -> list[dict[str, Any]]:
    try:
        stage3 = (
            stage3_context_bundle
            if isinstance(stage3_context_bundle, Stage3ContextualDecisionBundle)
            else Stage3ContextualDecisionBundle.model_validate(stage3_context_bundle)
        )
        stage4 = (
            stage4_context_bundle
            if isinstance(stage4_context_bundle, Stage4EvaluationBundle)
            else Stage4EvaluationBundle.model_validate(stage4_context_bundle)
        )
        stage5 = (
            stage5_diagnostic_bundle
            if isinstance(stage5_diagnostic_bundle, Stage5DiagnosticBundle)
            else Stage5DiagnosticBundle.model_validate(stage5_diagnostic_bundle)
        )
    except Exception as exc:
        raise Stage6ValidationError(f"Invalid Stage 6 upstream bundle: {exc}") from exc
    if len({stage3.case_id, stage4.case_id, stage5.case_id}) != 1:
        raise Stage6ValidationError("Stage 6 upstream case_id mismatch")

    evaluations = {item.decision_object_id: item for item in stage4.evaluations}
    diagnostics = {item.decision_object_id: item for item in stage5.scenario_diagnostics}
    object_ids = {item.decision_object_id for item in stage3.contextual_decision_objects}
    if object_ids != set(evaluations) or object_ids != set(diagnostics):
        raise Stage6ValidationError("Stage 6 requires exact Stage 3/4/5 decision-object alignment")
    frames = [
        build_stage6_synthesis_frame(
            decision_object=obj,
            evaluation=evaluations[obj.decision_object_id],
            scenario_diagnostic=diagnostics[obj.decision_object_id],
            system_evidence_diagnostic=stage5.system_evidence_diagnostic,
        )
        for obj in stage3.contextual_decision_objects
    ]
    frames.sort(
        key=lambda value: {"KR": 0, "EU": 1, "US": 2}[
            value["deterministic_output"]["jurisdiction_code"]
        ]
    )
    return frames


def build_stage6_bundle(
    *,
    case_id: str,
    frames: list[dict[str, Any]],
    narratives: list[SynthesisNarrative | dict[str, Any]],
) -> Stage6SynthesisBundle:
    if len(frames) != len(narratives):
        raise Stage6ValidationError("Stage 6 frames/narratives length mismatch")
    paired = [
        build_decision_synthesis(frame=frame, narrative=narrative)
        for frame, narrative in zip(frames, narratives, strict=True)
    ]
    syntheses = sorted(paired, key=lambda item: {"KR": 0, "EU": 1, "US": 2}[item.jurisdiction_code])
    return Stage6SynthesisBundle(
        case_id=case_id,
        synthesis_contract_version=STAGE6_SYNTHESIS_CONTRACT_VERSION,
        syntheses=syntheses,
    )

