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

    # Legacy Stage 2+ state. Preserved until those stages are redesigned.
    attack_plan: Optional[str]      # Current Round Attack Scenario
    defense_plan: Optional[str]     # Current Round Defense Strategy
    mediator_review: Optional[str]  # Mediator's Review and Decision Comment

    # Specialized agent analysis results
    technical_analysis: Optional[str]      # Technical Implementation Agent output
    regional_strategy: Optional[str]       # Regional Agent output
    finance_business_plan: Optional[str]   # Finance-Business Agent output

    iteration_count: int  # Loop Count (Preventing infinite loops)
    final_report: Optional[str]  # Final Output

    # Message history continues to accumulate using the "append" method.
    messages: Annotated[List[str], operator.add]    # History of messages/debate
