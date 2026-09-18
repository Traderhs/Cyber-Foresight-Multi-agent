import operator
from typing import Any, TypedDict, List, Optional, Annotated

class AgentState(TypedDict):
    """State Schema Definition for LangGraph"""
    stage0_config: Optional[dict[str, Any]]  # Explicit per-case config; env vars remain single-case fallback.
    forecast_data: str              # Stage 0 prompt-facing forecast/evidence view
    evidence_pack: Optional[dict[str, Any]]  # Stage 0 structured Agent input

    # Stage 1 independent pre-assessments. Both consume the same Stage 0 input and
    # neither is allowed to read the other's initial assessment.
    attack_assessment: Optional[dict[str, Any]]
    defense_assessment: Optional[dict[str, Any]]
    stage1_complete: bool
    stage1_artifact_reused: bool
    attack_checkpoint_reused: bool
    defense_checkpoint_reused: bool
    stage1_input_fingerprint: Optional[str]
    stage1_artifact_path: Optional[str]

    # Stage 2 bounded evidence-grounded debate. Stage 1 assessments remain the
    # immutable pre-assessments; post-assessments are stored separately.
    stage2_complete: bool
    stage2_artifact_reused: bool
    stage2_input_fingerprint: Optional[str]
    stage2_artifact_path: Optional[str]
    stage2_debate_result: Optional[dict[str, Any]]
    debate_rounds: List[dict[str, Any]]
    attack_post_assessment: Optional[dict[str, Any]]
    defense_post_assessment: Optional[dict[str, Any]]

    # Stage 3 deterministic Common Decision Object builder. This stage adds no
    # new LLM judgment; it freezes the shared evaluation unit for later lenses.
    stage3_complete: bool
    stage3_artifact_reused: bool
    stage3_input_fingerprint: Optional[str]
    stage3_artifact_path: Optional[str]
    stage3_decision_bundle: Optional[dict[str, Any]]
    decision_objects: List[dict[str, Any]]

    # Stage 3 contextualization v3. The original Stage 3 v2 DecisionObject is
    # preserved as a context-free baseline; this deterministic layer freezes
    # KR/EU/US CIS-IG2 scenarios plus Stage-4-specific decision evidence.
    stage3_context_complete: bool
    stage3_context_artifact_reused: bool
    stage3_context_input_fingerprint: Optional[str]
    stage3_context_artifact_path: Optional[str]
    stage3_contextual_decision_bundle: Optional[dict[str, Any]]
    contextual_decision_objects: List[dict[str, Any]]

    # Stage 4 information-isolated strategic value lenses. Each branch receives
    # the same frozen Stage 3 DecisionObject/evidence payload and cannot consume
    # sibling lens outputs from the same stage.
    stage4_complete: bool
    stage4_artifact_reused: bool
    stage4_input_fingerprint: Optional[str]
    stage4_artifact_path: Optional[str]
    stage4_evaluation_bundle: Optional[dict[str, Any]]
    technical_lens_assessments: List[dict[str, Any]]
    institutional_lens_assessments: List[dict[str, Any]]
    financial_lens_assessments: List[dict[str, Any]]
    technical_checkpoint_reuse_count: int
    institutional_checkpoint_reuse_count: int
    financial_checkpoint_reuse_count: int

    # Stage 4 contextual v2 outputs. Stage 4 v1 fields above remain available
    # solely for context-free baseline reproducibility.
    stage4_context_complete: bool
    stage4_context_artifact_reused: bool
    stage4_context_input_fingerprint: Optional[str]
    stage4_context_artifact_path: Optional[str]
    stage4_context_evaluation_bundle: Optional[dict[str, Any]]
    contextual_technical_lens_assessments: List[dict[str, Any]]
    contextual_institutional_lens_assessments: List[dict[str, Any]]
    contextual_financial_lens_assessments: List[dict[str, Any]]
    contextual_technical_checkpoint_reuse_count: int
    contextual_institutional_checkpoint_reuse_count: int
    contextual_financial_checkpoint_reuse_count: int

    # Stage 5 deterministic disagreement/evidence diagnostics. This layer makes
    # no LLM calls and never changes Stage 4 stances; it only computes metrics
    # and cross-jurisdiction sensitivity from frozen upstream artifacts.
    stage5_complete: bool
    stage5_artifact_reused: bool
    stage5_input_fingerprint: Optional[str]
    stage5_artifact_path: Optional[str]
    stage5_diagnostic_bundle: Optional[dict[str, Any]]

    # Stage 6 disagreement-preserving synthesis. A versioned deterministic
    # decision policy freezes the recommendation; the LLM only explains that
    # frozen policy result. Exactly two workers share the two-slot server.
    stage6_complete: bool
    stage6_artifact_reused: bool
    stage6_input_fingerprint: Optional[str]
    stage6_artifact_path: Optional[str]
    stage6_synthesis_bundle: Optional[dict[str, Any]]
    decision_syntheses: List[dict[str, Any]]
    stage6_narratives: List[dict[str, Any]]
    stage6_checkpoint_reuse_count: int

    # Stage 7 remains validation-harness-only.
    final_report: Optional[str]  # Legacy compatibility field; active Stage 6 uses structured syntheses.

    # Message history continues to accumulate using the "append" method.
    messages: Annotated[List[str], operator.add]    # History of messages/debate
