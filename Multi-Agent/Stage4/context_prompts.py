from __future__ import annotations

from Stage4.prompts import STAGE4_EVALUATION_USER_PROMPT, SYSTEM_PROMPT_BY_LENS


_CONTEXTUAL_RULES = r"""

Context-conditioned experiment rules:
11. For this contextual v4 experiment, the authoritative factual evidence boundary is the exact `evaluation_evidence`
   array in the payload, not only the original Stage 0 subset. It contains frozen Stage 0 forecast/operational
   records plus a frozen standard-derived context record and deterministic Stage-4 decision-evidence records.
   A `CTX_*` context record may support only facts about the standardized reference-enterprise profile. It cannot
   establish PMT effectiveness, implementation burden, cost, regulatory applicability, or legal obligation by itself.
12. The deployment context is a frozen experimental scenario. Treat its jurisdiction and reference-enterprise
    profile as fixed evaluation conditions, not as claims that a real observed organization has those properties.
13. Context fields do not replace factual evidence. External technical, legal, financial, or operational assertions
    still require exact supplied evidence IDs.
14. No exact employee count, staffing count, budget, price, ROI, TCO, payback period, vendor, product version,
    sector, or architecture may be inferred unless the supplied evidence explicitly supports it.
15. Compare only the action inside this scenario. Do not import facts or conclusions from another jurisdictional
    scenario or from another Stage 4 lens.
16. Any directional feasibility claim (`SUPPORTS_FEASIBILITY` or `CHALLENGES_FEASIBILITY`), whether GENERAL or
    DEPLOYMENT_SPECIFIC, must cite at least one action-specific Stage-4 decision-evidence record relevant to your
    lens. Forecast records and `CTX_*` profile records may provide context, but cannot by themselves justify a
    directional feasibility claim.
17. The payload contains only your assigned lens's deterministic `directional_grounding_contract`. Use that contract
    and the claim's scope. A decision-evidence record is directionally eligible only when it appears under that
    lens/scope contract. Do not treat mere topical relevance as directional grounding.
18. GENERAL and DEPLOYMENT_SPECIFIC are different questions. Missing information needed only for a narrower
    deployment-specific conclusion must not erase an otherwise grounded GENERAL direction. In that situation use
    a GENERAL directional claim, keep the missing deployment detail in evidence_gaps/conditional_requirements, and
    use PARTIAL rather than INSUFFICIENT_CONTEXT.
19. `stance=0` is for genuinely mixed directional evidence or for an indeterminate case with no surviving
    grounded direction. Do not use stance=0 merely because an exact budget, exact architecture, exact staffing
    count, exact legal applicability detail, or other narrower deployment-specific datum is unavailable. When
    evidence_sufficiency is PARTIAL or SUFFICIENT and stance=0, the claims must explicitly contain at least one
    grounded SUPPORTS_FEASIBILITY claim and at least one grounded CHALLENGES_FEASIBILITY claim. If no grounded
    directional claim survives, use INSUFFICIENT_EVIDENCE or INSUFFICIENT_CONTEXT instead of hiding a directional
    conclusion inside rationale/constraints while leaving all claims NEUTRAL_CONTEXT.
20. Conversely, missing context is never evidence for +1 or -1. A nonzero stance still requires at least one
    surviving directional claim grounded exactly as required by `directional_grounding_contract`.
21. The Stage 3 evidence contract uses the same predeclared retrieval procedure for support and challenge evidence,
    but it does NOT force equal evidence counts or equal weight. Explicitly consider both supplied directions before
    selecting a stance, then weigh them by directness to the frozen Threat×PMT action, empirical strength, source
    scope, deployment relevance, and documented limitations. Never use evidence-count voting. The mere presence of
    one support and one challenge record does not imply stance=0. A burden/constraint merely existing is not by
    itself proof that adoption is infeasible, and a benefit/implementation pathway merely existing is not by itself
    proof that adoption is feasible.
22. The `directional_grounding_contract` is direction-aware. SUPPORTS_FEASIBILITY claims must cite at least one
    direction-carrying record eligible under `support_slots`; CHALLENGES_FEASIBILITY claims must cite at least one
    direction-carrying record eligible under `challenge_slots`. Records that appear only under `eligible_slots`
    (for example deployment maturity) are supplementary context and cannot create direction by themselves. A record
    from the opposite facet may still be discussed neutrally, but it cannot be used to reverse its grounded direction.
23. When `required_scope_slots` is non-empty, the claim must also cite evidence covering every required scope slot.
    In particular, an Institutional/Regional DEPLOYMENT_SPECIFIC direction needs jurisdiction-specific
    regulatory_applicability in addition to a direction-carrying governance/compliance record.
24. Return the complete structured object every time. Explicitly include `claims`, `constraints`, `evidence_gaps`,
    and `conditional_requirements` arrays even when some of them are empty. Contextual v4 always supplies a
    non-empty action-specific evidence floor, so `claims` must contain at least one evidence-backed claim; do not
    omit the field during a semantic repair.
25. A jurisdiction label is a scenario parameter, not evidence. Do not change a GENERAL feasibility stance merely
    because the frozen region is KR, EU, or US. If your lens receives the same direction-carrying decision evidence
    and no jurisdiction-specific evidence eligible for that lens, the GENERAL weighing must remain jurisdiction-
    invariant. Put unresolved jurisdiction-specific questions in evidence_gaps / conditional_requirements instead.
    Only exact supplied jurisdiction-specific evidence may justify a region-driven change in feasibility direction.
26. If the payload region is `JURISDICTION_MASKED_NO_LENS_SPECIFIC_EVIDENCE`, this is an intentional
    jurisdiction-neutral equivalence-class evaluation, not a real region name. Do not infer, guess, or mention KR,
    EU, US, or any other jurisdiction. Evaluate only the supplied non-jurisdiction deployment context and evidence.
    Any narrower legal/regional conclusion must remain an evidence gap unless exact jurisdiction-specific evidence
    is present in the supplied grounding contract.
"""


_CONTEXTUAL_LENS_RULES = {
    "technical_feasibility": r"""

Contextual v4 Technical scope rules:
- GENERAL direction is evaluated primarily from exact Threat×PMT action evidence. Consider both direct
  technical_enablement and technical_limitation evidence, but do not treat them as equally strong merely because
  both are present. Generic PMT deployment guidance and deployment_maturity are supplementary context only.
- SUPPORTS_FEASIBILITY must be grounded by technical_enablement; CHALLENGES_FEASIBILITY must be grounded by
  technical_limitation. deployment_maturity may strengthen or qualify that judgment but cannot carry direction alone.
- DEPLOYMENT_SPECIFIC direction requires the same action-specific technical grounding plus the frozen context;
  do not infer compatibility with an unreported product/version/architecture detail.
- If evidence supports a general implementation/integration direction but an exact local architecture detail is
  missing, preserve the GENERAL direction and mark the assessment PARTIAL rather than forcing stance=0.
""",
    "institutional_regional": r"""

Contextual v4 Institutional / Regional scope rules:
- GENERAL institutional direction is evaluated from governance_enablement and compliance_constraint evidence, with
  regulatory_applicability used only when the supplied primary source actually applies to the frozen jurisdiction.
- SUPPORTS_FEASIBILITY must be grounded by governance_enablement; CHALLENGES_FEASIBILITY must be grounded by
  compliance_constraint. regulatory_applicability establishes jurisdictional scope but does not carry positive or
  negative feasibility direction by itself.
- DEPLOYMENT_SPECIFIC regional/legal direction requires both regulatory_applicability evidence for the frozen
  jurisdiction and the matching direction-carrying governance_enablement/compliance_constraint evidence. Generic
  governance/compliance material alone must not be promoted into a jurisdiction-specific legal claim, and an
  applicability record alone must not be interpreted as permission or prohibition.
- Do not turn absence of a directly applicable legal record into either support or challenge. Keep that narrower
  legal question in evidence_gaps while preserving any independently grounded GENERAL institutional direction.
""",
    "financial_adoption": r"""

Contextual v4 Financial / Adoption scope rules:
- GENERAL adoption direction evaluates the actually retrieved adoption_benefit and adoption_burden evidence without
  assuming equal counts or equal strength. deployment_maturity is supplementary context. Weigh direct economic or
  operational resource evidence more strongly than generic mentions of implementation activity.
- SUPPORTS_FEASIBILITY must be grounded by adoption_benefit; CHALLENGES_FEASIBILITY must be grounded by
  adoption_burden. deployment_maturity may strengthen or qualify the judgment but cannot carry direction alone.
- Exact affordability, budget fit, ROI, TCO, payback period, staffing count, and procurement timing remain
  unavailable unless explicitly evidenced. Their absence blocks those exact claims, not every directional judgment
  about adoption burden.
- A documented resource burden may contribute to a GENERAL adoption challenge without an exact dollar estimate, but
  it must be weighed against the supplied adoption-benefit and maturity evidence. Likewise, a documented efficiency
  benefit cannot erase documented staffing/integration burden. Never infer the sign from missing budget information.
""",
}


def _contextualize_base_prompt(value: str) -> str:
    value = value.replace(
        "You receive exactly one frozen DecisionObject and the exact Stage 0 evidence records listed in that object's\n"
        "evaluation_evidence_ids.",
        "You receive exactly one frozen contextual DecisionObject and the exact records listed in its\n"
        "evaluation_evidence_ids / evaluation_evidence payload.",
    )
    value = value.replace(
        "Factual claims must cite only exact Stage 0 evidence_id values from evaluation_evidence.",
        "Factual claims must cite only exact evidence_id values from the supplied evaluation_evidence array.",
    )
    return value


CONTEXT_SYSTEM_PROMPT_BY_LENS = {
    key: _contextualize_base_prompt(value) + _CONTEXTUAL_RULES + _CONTEXTUAL_LENS_RULES[key]
    for key, value in SYSTEM_PROMPT_BY_LENS.items()
}


CONTEXT_STAGE4_EVALUATION_USER_PROMPT = STAGE4_EVALUATION_USER_PROMPT

