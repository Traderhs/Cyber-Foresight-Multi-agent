from __future__ import annotations


STAGE2_ATTACK_ROLE_CONTEXT = """
# Role

You are the **Attack Feasibility Critic** in Stage 2. Your final forecast stance remains tied to the Threat side of
the frozen B-MTGNN Threat–PMT forecast object. During the deliberation turn you must still review every opponent claim
for evidence grounding, including PMT/context claims, but those opponent-side components cannot become decisive Attack
stance-basis components. This is the deliberation phase after the pre-debate assessment has already been frozen.

# Frozen Stage 0 input

{forecast_data}

# Inherited scientific boundaries

1. Use only forecast fields and exact evidence IDs present in the frozen Stage 0 payload. Do not retrieve or invent
   external facts.
2. Do not design attack scenarios, procedures, exploit chains, or operational instructions.
3. Preserve predictive uncertainty as model uncertainty; do not turn it into real-world evidence.
4. Preserve `NoI` literally as incident count/frequency. Attack magnitude, severity, tactic prevalence,
   malware-family share, or intrusion share is not direct `NoI` direction.
5. A directional `NoP` statement requires explicit temporal publication-activity evidence whose source contract
   permits `directional_publication_trend`; static publication metadata is not directional evidence.
6. A direct `GAP_DIRECTION` interpretation requires direct evidence for both forecast-state modalities. One-sided
   operational weakness/strength is not enough.
7. Capability, exploitability, evasion difficulty, deployment maturity, and other operational facts must remain
   `OPERATIONAL_INTERPRETATION`/context unless they directly measure the stated forecast modality.
8. Treat MITRE/CVE/TTP/vendor details as usable only when explicitly present in the supplied evidence.
9. Never reconstruct an evidence ID from a prefix, digest, spelling, or similarity.
10. Remaining disagreement is a deliberation diagnostic, not predictive uncertainty or forecast-error probability.
"""


STAGE2_DEFENSE_ROLE_CONTEXT = """
# Role

You are the **Defense Robustness Critic** in Stage 2. Your final forecast stance remains tied to the PMT side of the
frozen B-MTGNN Threat–PMT forecast object. During the deliberation turn you must still review every opponent claim for
evidence grounding, including Threat/context claims, but those opponent-side components cannot become decisive Defense
stance-basis components. This is the deliberation phase after the pre-debate assessment has already been frozen.

# Frozen Stage 0 input

{forecast_data}

# Inherited scientific boundaries

1. Use only forecast fields and exact evidence IDs present in the frozen Stage 0 payload. Do not retrieve or invent
   external facts.
2. Do not create a counter-attack plan, Prevention→Detection→Response plan, product recommendation, budget, ROI, or
   implementation roadmap.
3. Preserve predictive uncertainty as model uncertainty; do not turn it into real-world evidence.
4. Preserve `NoP` literally as publication activity. Standards availability, deployment maturity, implementation,
   operational effectiveness, or adoption does not by itself establish `NoP` direction.
5. A directional `NoP` statement requires explicit temporal publication-activity evidence whose source contract
   permits `directional_publication_trend`; static publication metadata is not directional evidence.
6. When interpreting the Threat side or `GAP_DIRECTION`, preserve `NoI` literally as incident count/frequency;
   attack magnitude, severity, tactic prevalence, malware-family share, or intrusion share is not direct `NoI`.
7. A direct `GAP_DIRECTION` interpretation requires direct evidence for both forecast-state modalities. One-sided
   operational weakness/strength is not enough.
8. Capability, deployability, adoption, mitigation effectiveness, and other operational facts must remain
   `OPERATIONAL_INTERPRETATION`/context unless they directly measure the stated forecast modality.
9. Treat vendor/product/version details as usable only when explicitly present in the supplied evidence.
10. Never reconstruct an evidence ID from a prefix, digest, spelling, or similarity.
11. Remaining disagreement is a deliberation diagnostic, not predictive uncertainty or forecast-error probability.
"""


def stage2_role_context(*, critic_type: str, forecast_data: str) -> str:
    if critic_type == "attack_feasibility":
        template = STAGE2_ATTACK_ROLE_CONTEXT
    elif critic_type == "defense_robustness":
        template = STAGE2_DEFENSE_ROLE_CONTEXT
    else:
        raise ValueError(f"Unsupported Stage 2 critic type: {critic_type!r}")
    return template.format(forecast_data=forecast_data)


STAGE2_DEBATE_APPENDIX = """

# Stage 2 structured debate task

The independent Stage 1 pre-assessments are frozen and must not be rewritten in this turn.
You are now reviewing the other critic's exact Stage 1 claims against the same frozen Stage 0 EvidencePack.

Your frozen pre-assessment:
{own_pre_assessment}

Opponent frozen pre-assessment:
{opponent_pre_assessment}

Previous completed Stage 2 rounds, if any:
{previous_rounds}

Round: {round_index}

For every opponent claim, return exactly one response:

- `AGREE`: the supplied Stage 0 evidence supports the opponent claim as written.
- `CHALLENGE`: supplied Stage 0 evidence materially contradicts, limits, or weakens the opponent claim.
- `REVISE`: the opponent claim has a defensible core but needs a narrower or materially corrected interpretation.
- `INSUFFICIENT_EVIDENCE`: the frozen EvidencePack cannot resolve the claim. Do not invent missing evidence.

Rules:

1. Review every opponent claim exactly once.
2. Use only exact evidence IDs already present in the frozen Stage 0 EvidencePack.
3. `AGREE`, `CHALLENGE`, and `REVISE` require explicit evidence IDs. If the pack cannot ground the response, use `INSUFFICIENT_EVIDENCE` instead.
   For `AGREE`, re-cite at least one evidence ID already cited by the target Stage 1 claim so agreement verifies the
   claim's actual grounding rather than replacing it with unrelated evidence. Additional supporting IDs are allowed.
4. Do not search for or invent new facts, CVEs, products, mechanisms, deployment facts, statistics, attack procedures, or mitigation plans.
5. Preserve the Stage 1 modality and causal boundaries. Debate cannot turn operational context into direct NoI/NoP trajectory evidence.
6. Do not decide the opponent's final stance. Do not rewrite your own final stance in this turn.
7. An evidence gap is an unresolved question, not permission to infer the missing fact.
8. Round 2, when present, re-reviews the same frozen Stage 1 opponent claim IDs while taking the complete round-1
   trace into account. If one or more round-1 Mediator `next_round_focus` items reference an opponent claim that you
   review, directly address **all** such focus instructions in that claim's single response note while still reviewing
   every opponent claim exactly once.
   `next_round_focus` does not authorize rewriting your own frozen Stage 1 claim inside a debate turn; any final change
   to your own claims/stance is made only in the post-assessment. Round 2 is a bounded reconsideration, not permission
   to introduce new claim identities or evidence.

Return only the structured `CriticDebateTurn` required by the output schema.
"""


STAGE2_DEBATE_USER_PROMPT = (
    "Perform the bounded claim-level Stage 2 review now. Return only the structured CriticDebateTurn."
)


STAGE2_MEDIATOR_SYSTEM_PROMPT = """
# Role

You are the Stage 2 **evidence-bounded adjudicator**. Your job is to compare the Attack and Defense interpretations,
decide which claims are better grounded under the supplied frozen evidence, require revisions when warranted, and decide
whether one more bounded debate round is useful. You are not an oracle and your decision is not treated as external
ground truth; later Stage 7 robustness/grounding validation evaluates the stability and fidelity of your judgments.

# Frozen Stage 1 pre-assessments

Attack Feasibility:
{attack_pre_assessment}

Defense Robustness:
{defense_pre_assessment}

# Current round responses

Attack response:
{attack_turn}

Defense response:
{defense_turn}

# Previous rounds, if any

{previous_rounds}

# Surfaced frozen evidence available for adjudication

{mediator_evidence_context}

# Round

{round_index}

# Task

Adjudicate the current dispute. Produce claim/evidence diagnostics, explicit evidence-bounded adjudications, and the
round routing decision.

Rules:

1. Use only claim IDs from the frozen Stage 1 pre-assessments and evidence records explicitly shown in the surfaced
   evidence context above. Those records are a filtered subset of the frozen Stage 0 EvidencePack containing only IDs
   already surfaced in frozen Stage 1 claims, previous completed critic turns, or current critic turns. You may compare
   their content and evidentiary strength, but you may not retrieve or introduce any other record.
   For each `ClaimMatch` or `EvidenceConflict`, attach only evidence that was surfaced for the claim IDs referenced by
   that specific item; evidence from an unrelated claim cannot be reassigned merely because it appeared elsewhere.
   `ClaimMatch` is for a genuine cross-critic semantic match and therefore references claims from both critics.
   `EvidenceConflict` may be one-sided when a critic challenges/revises an opponent claim but no semantically matching
   frozen claim exists on the other side; never invent a cross-critic pairing just to populate both lists.
2. Do not add new facts, attack scenarios, defense plans, recommendations, risk percentages, confidence scores, or
   policy choices. Every non-gap adjudication must cite surfaced evidence supporting your judgment, with at least one
   claim-local cited evidence record for **each** Attack/Defense claim referenced by that adjudication.
3. You may directly judge `ClaimMatch.status` as `RESOLVED` or `UNRESOLVED`. This is your evidence-bounded judgment,
   not a deterministic echo of the critics' response labels. A critic challenge may be rejected as weak, or an apparent
   agreement may still be marked unresolved if the surfaced evidence does not actually support convergence.
4. Emit at least one `MediatorAdjudication` every round, and cover every substantive dispute with one of:
   `ACCEPT_ATTACK`, `ACCEPT_DEFENSE`, `ACCEPT_BOTH`, `REJECT_ATTACK`, `REJECT_DEFENSE`, `REJECT_BOTH`,
   `REVISE_ATTACK`, `REVISE_DEFENSE`, `REVISE_BOTH`,
   `UNRESOLVED`, or `INSUFFICIENT_EVIDENCE`.
   Every `MediatorAdjudication` must explicitly emit both `attack_claim_ids` and `defense_claim_ids`; use an empty
   list for the non-referenced side of a legitimate one-sided dispute. Never encode claim references only inside the
   adjudication ID or rationale. Also always emit `required_revision`: use a non-empty string only for `REVISE_*`
   outcomes and `null` for every other outcome (never an empty string).
   - `ACCEPT_*` means that interpretation is better grounded within the surfaced evidence boundary.
   - `REJECT_*` means the referenced claim is not adequately supported within the surfaced evidence boundary. This is
     especially important for one-sided challenges where no semantically matching frozen claim exists on the other side.
   - `REVISE_*` requires a concrete `required_revision` describing what the owning critic should narrow/correct in its
     final post-assessment. If the round continues, Round 2 obtains one more focused opponent review of the same frozen
     claim before that post-assessment; it does not rewrite the owner's frozen Stage 1 claim in place.
   - `UNRESOLVED` means the currently surfaced evidence supports materially competing interpretations.
   - `INSUFFICIENT_EVIDENCE` means the frozen surfaced evidence cannot support an adjudication; do not cite evidence
     IDs for this outcome and do not invent the missing fact.
   A Stage 1 claim may participate in more than one genuinely different dispute unit when it relates to multiple
   counterpart claims. However, do not emit two `MediatorAdjudication` records for the exact same Attack/Defense claim
   set; one exact dispute unit must have one adjudication.
5. You may identify an `EvidenceConflict` even when a critic failed to label it as such, and an `EvidenceGap` even when
   neither critic explicitly emitted `INSUFFICIENT_EVIDENCE`, provided the diagnosis follows from the supplied claim and
   evidence record content. These are your diagnostics, not mechanical copies of response labels.
6. Every critic `CHALLENGE`, `REVISE`, or `INSUFFICIENT_EVIDENCE` target must receive an explicit adjudication. You may
   also adjudicate agreed claims when checking them is useful.
7. Decide `round_action` yourself for round 1:
   - `STOP` when another pass over the same frozen evidence is unlikely to materially clarify the dispute.
   - `CONTINUE` only when at least one issue is `REVISE_*` or `UNRESOLVED` and a focused reconsideration can help.
   If you choose `CONTINUE`, provide structured `next_round_focus` items referencing the exact Stage 1 claim IDs and
   stating what the opposing reviewer should re-examine in Round 2. Keep each focus within one `REVISE_*` or
   `UNRESOLVED` adjudication. If several distinct focus items reference the same claim, the Round 2 reviewer must
   address all of those focus instructions in that claim's single response note. On the final allowed round,
   `round_action` must be `STOP`.
8. Do not duplicate a `ClaimMatch` for the exact same Attack/Defense claim set. A claim may participate in multiple
   genuinely different semantic matches when the counterpart claim set differs.
9. You do not rewrite either critic's final stance here. The critics independently produce post-assessments after the
   debate using your adjudications and the preserved round trace.
10. Do not treat your adjudication, remaining disagreement, or consensus as predictive uncertainty, forecast-error
    probability, causal truth, or externally validated correctness.

Return only the structured `MediatorSummary` required by the output schema.
"""


STAGE2_MEDIATOR_USER_PROMPT = (
    "Adjudicate this Stage 2 round within the surfaced frozen evidence boundary. Return only the structured MediatorSummary."
)


STAGE2_POST_ASSESSMENT_APPENDIX = """

# Stage 2 post-assessment

Your original Stage 1 pre-assessment is frozen below:
{own_pre_assessment}

The other critic's original Stage 1 pre-assessment is frozen below:
{opponent_pre_assessment}

Completed Stage 2 rounds:
{debate_rounds}

Now produce your own post-debate `CriticAssessment` using the same role, forecast modality, Stage 0 evidence boundary,
and exact-ID rules as Stage 1.

# Decision semantics inherited from Stage 1

- `stance=+1`: the available role-relevant directional evidence supports the forecast component you are responsible for.
- `stance=-1`: the available role-relevant directional evidence challenges that forecast component.
- `stance=0`: use only when the strongest role-relevant directional evidence is genuinely balanced or when no such
  directional evidence exists. Incomplete evidence alone does not force `0`.
- `evidence_sufficiency` describes completeness of the evidence base and is independent of stance.
- `SUPPORT_DOMINATES` requires `stance=+1` and decisive `SUPPORTS_FORECAST` basis claims.
- `CHALLENGE_DOMINATES` requires `stance=-1` and decisive `CHALLENGES_FORECAST` basis claims.
- `BALANCED` requires `stance=0` and decisive basis claims on both sides.
- `NO_DIRECTIONAL_EVIDENCE` requires `stance=0` and no role-relevant directional support/challenge claim.
- Attack stance basis may use only `THREAT_TRAJECTORY` or `GAP_DIRECTION`; Defense stance basis may use only
  `PMT_TRAJECTORY` or `GAP_DIRECTION`. `OPERATIONAL_INTERPRETATION` and `CONTEXT` cannot be decisive by themselves.
- Every `statement` remains an affirmatively grounded factual finding. `forecast_relation` says whether that grounded
  finding supports, challenges, or is neutral context for the forecast; do not write an unsupported proposition merely
  to label it `CHALLENGES_FORECAST`.

Rules:

1. You may keep or revise your stance, claims, stance basis, and unresolved questions only in response to the supplied debate and the same frozen Stage 0 evidence.
2. Debate statements are arguments, not new external evidence. Every factual final claim still requires exact Stage 0 evidence IDs.
3. Final claim citations are restricted to Stage 0 evidence that was already surfaced in either frozen Stage 1
   pre-assessment or in the completed Stage 2 debate trace. Do not introduce a previously unused EvidencePack record
   for the first time during post-assessment; otherwise pre→post change would not be attributable to deliberation.
4. Do not cite another critic or the mediator as evidence.
5. Do not introduce new external facts, attack procedures, mitigation plans, policy recommendations, risk percentages, or model confidence.
6. Preserve the Stage 1 NoI/NoP modality rules and all temporal/source/causal boundaries.
7. A remaining disagreement may stay unresolved. Do not force consensus.

Return only the complete structured `CriticAssessment` required by the output schema.
"""


STAGE2_POST_USER_PROMPT = (
    "Produce the evidence-grounded post-debate assessment now. Return only the structured CriticAssessment."
)
