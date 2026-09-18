from __future__ import annotations

import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from Stage4.schema import LensAssessment, LensClaimStatus, LensEvidenceSufficiency, LensType


STAGE6_SYNTHESIS_CONTRACT_VERSION = "stage6-synthesis-v5"
STAGE6_SEMANTIC_VALIDATION_VERSION = "stage6-semantic-validation-v12"
STAGE6_DECISION_POLICY_VERSION = "stage6-decision-policy-v1"
STAGE6_SYNTHESIS_AUTHORITY = "DETERMINISTIC_POLICY_WITH_EVIDENCE_GROUNDED_LLM_REPORT"
STAGE6_RECOMMENDATION_INTERPRETATION = "POLICY_CONDITIONAL_DECISION_SUPPORT"
STAGE6_LENS_SET_ROLE = "PREDECLARED_FUNCTIONAL_DIMENSIONS_NOT_STATISTICAL_SAMPLE"
STAGE6_D_LENS_INTERPRETATION = "DESCRIPTIVE_WITHIN_SET_DISPERSION_NOT_POPULATION_UNCERTAINTY"


class Stage6ValidationError(ValueError):
    """Raised when a Stage 6 synthesis violates the frozen synthesis contract."""


_NUMERIC_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_])[+-]?\d+(?:[.,]\d+)*(?:\s*(?:%|percent(?:age)?(?:\s+points?)?))?",
    flags=re.IGNORECASE,
)


def normalize_numeric_token(value: str) -> str:
    """Canonicalize equivalent numeric renderings used in grounded prose.

    This intentionally treats ``21%`` and ``21 percent`` as the same source
    quantity while keeping percentage points distinct from percentages.
    """

    compact = " ".join(str(value).strip().split()).casefold()
    suffix = ""
    if compact.endswith("percentage point") or compact.endswith("percentage points"):
        suffix = "pp"
        compact = re.sub(r"\s*percentage\s+points?$", "", compact)
    elif compact.endswith("percent") or compact.endswith("percentage") or compact.endswith("%"):
        suffix = "%"
        compact = re.sub(r"\s*(?:percent(?:age)?|%)$", "", compact)
    compact = compact.replace(",", "")
    if compact.startswith("+"):
        compact = compact[1:]
    return compact + suffix


def extract_normalized_numeric_tokens(value: str) -> set[str]:
    return {
        normalize_numeric_token(match.group(0))
        for match in _NUMERIC_TOKEN_RE.finditer(str(value))
    }


class DecisionRecommendation(StrEnum):
    """Deterministic Stage 6 action recommendation for the frozen candidate action."""

    RECOMMEND = "RECOMMEND"
    RECOMMEND_PILOT = "RECOMMEND_PILOT"
    HOLD = "HOLD"
    DO_NOT_RECOMMEND = "DO_NOT_RECOMMEND"


class SynthesisGap(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gap_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    text: str = Field(min_length=1)


class SynthesisCondition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    condition_id: str = Field(min_length=1)
    source_lens: LensType
    text: str = Field(min_length=1)


class EvidenceQualitySummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_sufficiency_by_lens: dict[str, LensEvidenceSufficiency]
    d_lens_evaluable: bool
    joint_abstention: bool
    agreement_class: str = Field(min_length=1)
    stage0_evidence_slot_coverage_ratio: float = Field(ge=0.0, le=1.0)
    source_family_count: int = Field(ge=0)
    claim_evidence_reference_pass_rate: float = Field(ge=0.0, le=1.0)
    semantic_grounding_audit_status: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_lens_keys(self) -> "EvidenceQualitySummary":
        if set(self.evidence_sufficiency_by_lens) != {lens.value for lens in LensType}:
            raise ValueError("evidence_sufficiency_by_lens must contain exactly the three Stage 4 lenses")
        return self


class SynthesisNarrative(BaseModel):
    """The only model-generated portion of Stage 6.

    Every structural decision field, including the recommendation and the full
    Stage 4 evidence trace, is assembled deterministically before the model is
    called. The model writes a human-facing strategic-intelligence report from
    that frozen material and explicitly cites the evidence IDs it relies on.
    """

    model_config = ConfigDict(extra="forbid")

    decision_object_id: str = Field(min_length=1)
    decision_recommendation: DecisionRecommendation
    strategic_intelligence_report: str = Field(min_length=300, max_length=16000)


class DecisionSynthesis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision_object_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    jurisdiction_code: str = Field(min_length=1)
    candidate_action_id: str = Field(min_length=1)
    candidate_action_name: str = Field(min_length=1)
    forecast_summary: dict[str, Any]
    predictive_uncertainty_summary: dict[str, Any]
    critique_summary: dict[str, Any]
    supporting_lenses: list[LensType]
    dissenting_lenses: list[LensType]
    indeterminate_lenses: list[LensType]
    decision_recommendation: DecisionRecommendation
    decision_policy_version: str = Field(min_length=1)
    decision_rule_id: str = Field(min_length=1)
    recommendation_scope: str = Field(min_length=1)
    recommendation_interpretation: str = Field(min_length=1)
    lens_set_role: str = Field(min_length=1)
    d_lens_interpretation: str = Field(min_length=1)
    d_lens: float = Field(ge=0.0, le=1.0)
    evidence_quality_summary: EvidenceQualitySummary
    lens_evidence_trace: dict[str, LensAssessment]
    unresolved_evidence_gaps: list[SynthesisGap]
    strategic_intelligence_report: str = Field(min_length=300, max_length=16000)
    report_cited_evidence_ids: list[str] = Field(min_length=1)
    upstream_conditions: list[SynthesisCondition]
    synthesis_authority: str = Field(min_length=1)
    non_claims: list[str] = Field(min_length=1)
    provenance: dict[str, Any]

    @model_validator(mode="after")
    def validate_lens_partition(self) -> "DecisionSynthesis":
        groups = [self.supporting_lenses, self.dissenting_lenses, self.indeterminate_lenses]
        expected = set(LensType)
        combined = [lens for group in groups for lens in group]
        if len(combined) != len(set(combined)) or set(combined) != expected:
            raise ValueError("supporting/dissenting/indeterminate lenses must partition all three lenses")
        for values in groups:
            if values != sorted(values, key=lambda item: item.value):
                raise ValueError("lens lists must be deterministically sorted")
        gap_ids = [item.gap_id for item in self.unresolved_evidence_gaps]
        condition_ids = [item.condition_id for item in self.upstream_conditions]
        if len(gap_ids) != len(set(gap_ids)):
            raise ValueError("unresolved_evidence_gaps IDs must be unique")
        if len(condition_ids) != len(set(condition_ids)):
            raise ValueError("upstream_conditions IDs must be unique")
        if len(self.non_claims) != len(set(self.non_claims)):
            raise ValueError("non_claims must be unique")
        if self.synthesis_authority != STAGE6_SYNTHESIS_AUTHORITY:
            raise ValueError("Stage 6 synthesis authority mismatch")
        if self.decision_policy_version != STAGE6_DECISION_POLICY_VERSION:
            raise ValueError("Stage 6 decision policy version mismatch")
        if self.recommendation_interpretation != STAGE6_RECOMMENDATION_INTERPRETATION:
            raise ValueError("Stage 6 recommendation interpretation mismatch")
        if self.lens_set_role != STAGE6_LENS_SET_ROLE:
            raise ValueError("Stage 6 lens-set role mismatch")
        if self.d_lens_interpretation != STAGE6_D_LENS_INTERPRETATION:
            raise ValueError("Stage 6 D_lens interpretation mismatch")
        if set(self.lens_evidence_trace) != {lens.value for lens in LensType}:
            raise ValueError("lens_evidence_trace must contain exactly the three Stage 4 lenses")
        for lens in LensType:
            assessment = self.lens_evidence_trace[lens.value]
            if assessment.lens_type != lens:
                raise ValueError(f"lens_evidence_trace lens mismatch for {lens.value}")
            if assessment.decision_object_id != self.decision_object_id:
                raise ValueError(f"lens_evidence_trace decision_object_id mismatch for {lens.value}")
        if len(self.report_cited_evidence_ids) != len(set(self.report_cited_evidence_ids)):
            raise ValueError("report_cited_evidence_ids must be unique")
        return self


class Stage6SynthesisBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    synthesis_contract_version: str = Field(min_length=1)
    syntheses: list[DecisionSynthesis] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def validate_bundle(self) -> "Stage6SynthesisBundle":
        if self.synthesis_contract_version != STAGE6_SYNTHESIS_CONTRACT_VERSION:
            raise ValueError("Stage 6 synthesis contract version mismatch")
        scenarios = [item.scenario_id for item in self.syntheses]
        if set(scenarios) != {"KR__CIS_IG2", "EU__CIS_IG2", "US__CIS_IG2"}:
            raise ValueError("Stage 6 main bundle must contain exactly KR/EU/US CIS-IG2 syntheses")
        if len(scenarios) != len(set(scenarios)):
            raise ValueError("Stage 6 scenario syntheses must be unique")
        return self


_FORBIDDEN_SUMMARY_PATTERNS = (
    # The deterministic recommendation is authoritative. The LLM may explain
    # it but may not promote it into unsupported deployment guarantees or a
    # different aggregation theory.
    r"\bdeploy\s+(?:now|immediately)\b",
    r"\bfull\s+production\s+deployment\b",
    r"\bfinal\s+stance\b",
    r"\bmajority\s+(?:determines?|decides?|wins?|controls?|selects?|sets?)\b",
    r"\bmajority\s+vote\s+(?:determines?|decides?|wins?|controls?|selects?|sets?)\b",
    r"\b(?:decision|recommendation|outcome)\s+(?:is|was)\s+based\s+on\s+(?:a\s+)?majority\b",
    r"\bwinner\b",
    r"\bloser\b",
    r"\baverage\s+stance\b",
    r"\boptimal\s+policy\b",
    r"\bcausally\s+best\b",
    r"\bguarantees?\s+(?:utility|deployment|success)\b",
)

_DRAFT_DEBUG_RESIDUE_PATTERNS = (
    r"\bplaceholder\b",
    r"\btodo\b",
    r"\bfixme\b",
    r"\btypo\b",
    r"\btest(?:ing)?\s+(?:strictness|schema|validation)\b",
    r"\bcorrect\s+(?:key|field|value|schema)\b",
    r"\bneed\s+(?:the\s+)?correct\s+(?:key|field|value|schema)\b",
    r"\bthis\s+field\b",
)


def _lens_assessments_from_frame(frame: dict[str, Any]) -> dict[str, dict[str, Any]]:
    values = frame.get("deterministic_output", {}).get("lens_evidence_trace")
    if not isinstance(values, dict):
        raise Stage6ValidationError("Stage 6 frame is missing lens_evidence_trace")
    expected = {lens.value for lens in LensType}
    if set(values) != expected:
        raise Stage6ValidationError("Stage 6 frame lens_evidence_trace is incomplete")
    return values


def _claim_evidence_ids(claims: list[dict[str, Any]], *, status: str | None = None) -> set[str]:
    return {
        str(evidence_id)
        for claim in claims
        if status is None or str(claim.get("status")) == status
        for evidence_id in claim.get("evidence_ids", [])
        if evidence_id
    }


_INLINE_EVIDENCE_BRACKET_RE = re.compile(r"\[([^\]]*EVIDENCE:[^\]]*)\]")
_INLINE_EVIDENCE_GROUP_CONTENT_RE = re.compile(
    r"EVIDENCE:[^\s,;\]]+(?:\s*,\s*EVIDENCE:[^\s,;\]]+)*"
)
_INLINE_EVIDENCE_TOKEN_RE = re.compile(r"(?:^|\s*,\s*)EVIDENCE:([^\s,;\]]+)")


def _inline_report_evidence_ids(report: str) -> set[str]:
    """Extract exact evidence IDs from inline evidence citation brackets.

    The canonical form remains one citation per bracket, for example
    ``[EVIDENCE:E1]``.  In practice a model may compact adjacent citations into
    one bracket such as ``[EVIDENCE:E1, EVIDENCE:E2]``.  Treat that harmless
    presentation variant as the same two exact citations rather than falsely
    reporting missing lens evidence.  Only explicit ``EVIDENCE:`` tokens inside
    square brackets are recognized; bare comma-separated IDs are not inferred.
    Unknown extracted IDs are still rejected against the frozen upstream
    evidence universe by the caller.
    """

    cited: set[str] = set()
    for bracket in _INLINE_EVIDENCE_BRACKET_RE.finditer(report):
        content = bracket.group(1)
        if _INLINE_EVIDENCE_GROUP_CONTENT_RE.fullmatch(content) is None:
            continue
        cited.update(match.group(1) for match in _INLINE_EVIDENCE_TOKEN_RE.finditer(content))
    return cited


def _malformed_inline_evidence_brackets(report: str) -> list[str]:
    return [
        bracket.group(0)
        for bracket in _INLINE_EVIDENCE_BRACKET_RE.finditer(report)
        if _INLINE_EVIDENCE_GROUP_CONTENT_RE.fullmatch(bracket.group(1)) is None
    ]


def _allowed_report_evidence_ids(frame: dict[str, Any]) -> set[str]:
    """Return the complete frozen evidence universe available to Stage 6.

    Contextual Stage 3 guarantees that evaluation_evidence_ids is exactly the
    union of forecast, context, and decision evidence. Stage 4 claim evidence
    is a directional subset of that universe, so a final report may cite
    upstream forecast/context evidence without pretending that it was a
    directional Stage 4 lens claim.
    """

    provenance = frame.get("deterministic_output", {}).get("provenance") or {}
    values = provenance.get("evaluation_evidence_ids")
    if not isinstance(values, list) or not values:
        raise Stage6ValidationError(
            "Stage 6 frame is missing the frozen evaluation_evidence_ids universe"
        )
    if any(not isinstance(item, str) or not item for item in values):
        raise Stage6ValidationError("Stage 6 evaluation_evidence_ids must be non-empty strings")
    if len(values) != len(set(values)):
        raise Stage6ValidationError("Stage 6 evaluation_evidence_ids must be unique")
    return set(values)


def extract_report_evidence_ids(*, report: str, frame: dict[str, Any]) -> list[str]:
    """Derive report citations from the frozen upstream evidence universe.

    The LLM writes only the human-facing report. Citation bookkeeping is
    deterministic so a separately generated evidence-ID array cannot drift,
    duplicate, or enter an unbounded repetition loop.
    """

    allowed = _allowed_report_evidence_ids(frame)
    cited = _inline_report_evidence_ids(report)
    return sorted(cited & allowed)


def validate_synthesis_narrative(
    narrative: SynthesisNarrative | dict[str, Any],
    *,
    frame: dict[str, Any],
) -> SynthesisNarrative:
    try:
        parsed = (
            narrative
            if isinstance(narrative, SynthesisNarrative)
            else SynthesisNarrative.model_validate(narrative)
        )
    except Exception as exc:
        raise Stage6ValidationError(f"Invalid Stage 6 narrative schema: {exc}") from exc

    frozen = frame["deterministic_output"]
    errors: list[str] = []
    if parsed.decision_object_id != frozen["decision_object_id"]:
        errors.append("decision_object_id changed")
    if parsed.decision_recommendation.value != frozen["decision_recommendation"]:
        errors.append("decision recommendation changed")

    text = parsed.strategic_intelligence_report
    lowered = text.casefold()
    for pattern in _FORBIDDEN_SUMMARY_PATTERNS:
        if re.search(pattern, lowered):
            errors.append(f"decision-support summary used forbidden directive/meta language: {pattern}")

    basis = frame.get("narrative_source", {}).get("decision_basis_contract") or {}
    if basis.get("trigger_logic") == "ANY_LISTED_TRIGGER_IS_POLICY_SUFFICIENT":
        # Under the asymmetric critical-gate policy, one listed trigger may be
        # sufficient to determine the policy output. Preserve that distinction
        # while still allowing caveats about universal validity.
        single_trigger_negation_patterns = (
            r"\b(?:no|not)\s+(?:single|one)\s+(?:lens|dimension|gate)\s+(?:is|can\s+be)\s+decisive\b",
            r"\bdoes\s+not\s+imply\s+that\s+(?:a|one)\s+(?:single\s+)?(?:lens|dimension|gate)\s+alone\s+is\s+decisive\b",
        )
        for pattern in single_trigger_negation_patterns:
            if re.search(pattern, lowered):
                errors.append(
                    "strategic-intelligence report contradicted the frozen critical-gate trigger logic: "
                    f"{pattern}"
                )

    # Reject clear draft/debug residue without imposing any substantive answer
    # template. A single generic word is not enough; multiple independent
    # editing/debug signals must co-occur before the narrative is rejected.
    draft_debug_hits = [
        pattern for pattern in _DRAFT_DEBUG_RESIDUE_PATTERNS if re.search(pattern, lowered)
    ]
    if len(draft_debug_hits) >= 2:
        errors.append(
            "decision-support summary appears to contain draft/debug/placeholder residue: "
            f"{draft_debug_hits}"
        )

    lens_assessments = _lens_assessments_from_frame(frame)
    malformed_citations = _malformed_inline_evidence_brackets(text)
    if malformed_citations:
        errors.append(
            "strategic-intelligence report used malformed inline evidence citation syntax: "
            f"{malformed_citations}"
        )
    inline_citations = _inline_report_evidence_ids(text)
    allowed_report_evidence_ids = _allowed_report_evidence_ids(frame)
    outside = sorted(inline_citations - allowed_report_evidence_ids)
    if outside:
        errors.append(
            "strategic-intelligence report cited evidence outside the frozen upstream evidence universe: "
            f"{outside}"
        )
    cited = inline_citations & allowed_report_evidence_ids

    for lens in LensType:
        assessment = lens_assessments[lens.value]
        claims = list(assessment.get("claims") or [])
        all_ids = _claim_evidence_ids(claims)
        cited_for_lens = cited & all_ids
        if claims and not cited_for_lens:
            errors.append(
                f"strategic-intelligence report omitted an evidence trace for {lens.value}"
            )

        stance = int(assessment.get("stance"))
        support_ids = _claim_evidence_ids(
            claims,
            status=LensClaimStatus.SUPPORTS_FEASIBILITY.value,
        )
        challenge_ids = _claim_evidence_ids(
            claims,
            status=LensClaimStatus.CHALLENGES_FEASIBILITY.value,
        )
        if stance == 1 and support_ids and not (cited & support_ids):
            errors.append(
                f"strategic-intelligence report omitted decisive support evidence for {lens.value}"
            )
        elif stance == -1 and challenge_ids and not (cited & challenge_ids):
            errors.append(
                f"strategic-intelligence report omitted decisive challenge evidence for {lens.value}"
            )
        elif stance == 0 and support_ids and challenge_ids:
            if not (cited & support_ids):
                errors.append(
                    f"strategic-intelligence report omitted mixed support evidence for {lens.value}"
                )
            if not (cited & challenge_ids):
                errors.append(
                    f"strategic-intelligence report omitted mixed challenge evidence for {lens.value}"
                )

    if not cited:
        errors.append("strategic-intelligence report did not cite any frozen upstream evidence ID inline")

    # The synthesis prose is not the place to invent quantitative claims.
    # Numbers are allowed when they already occur in the frozen synthesis
    # frame OR in the exact frozen evidence records that the report actually
    # cites inline.  This keeps the validator provenance-aware without opening
    # a case-wide numeric whitelist from uncited evidence.
    prompt_visible_frame = {
        key: value for key, value in frame.items() if key != "validation_context"
    }
    allowed_numbers = extract_normalized_numeric_tokens(str(prompt_visible_frame))
    numeric_by_evidence = (
        frame.get("validation_context", {}).get("evidence_numeric_tokens_by_id", {})
    )
    if not isinstance(numeric_by_evidence, dict):
        errors.append("Stage 6 frame numeric provenance is malformed")
        numeric_by_evidence = {}
    for evidence_id in cited:
        values = numeric_by_evidence.get(evidence_id, [])
        if isinstance(values, list):
            allowed_numbers.update(str(value) for value in values)
    generated_numbers = extract_normalized_numeric_tokens(text)
    new_numbers = sorted(generated_numbers - allowed_numbers)
    if new_numbers:
        errors.append(f"decision-support summary introduced unsupported numeric token(s): {new_numbers}")
    if errors:
        raise Stage6ValidationError("; ".join(errors))
    return parsed


def build_constrained_synthesis_narrative_schema(*, frame: dict[str, Any]) -> dict[str, Any]:
    schema = SynthesisNarrative.model_json_schema()
    frozen = frame["deterministic_output"]
    schema["properties"]["decision_object_id"] = {
        "type": "string",
        "const": frozen["decision_object_id"],
    }
    schema["properties"]["decision_recommendation"] = {
        "type": "string",
        "const": frozen["decision_recommendation"],
    }
    schema["required"] = sorted(schema["properties"])
    return schema

