import operator
from typing import TypedDict, List, Optional, Annotated

class AgentState(TypedDict):
    """State Schema Definition for LangGraph"""
    forecast_data: str              # B-MTGNN Forecast Data (Read-only Context)

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
