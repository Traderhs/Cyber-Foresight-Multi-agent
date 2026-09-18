from langgraph.graph import StateGraph, END
from Pipeline.state import AgentState
from Stage0.nodes import load_data_node
from Stage1.nodes import (
    prepare_stage1_node,
    attack_feasibility_critic_node,
    defense_robustness_critic_node,
    stage1_complete_node,
)
from Stage2.nodes import prepare_stage2_node, stage2_debate_node, stage2_complete_node
from Stage3.nodes import stage3_build_node
from Stage3.context_nodes import stage3_contextualize_node
from Stage4.nodes import (
    prepare_stage4_node,
    technical_feasibility_lens_node,
    institutional_regional_lens_node,
    financial_adoption_lens_node,
    stage4_complete_node,
)
from Stage4.context_nodes import (
    prepare_stage4_context_node,
    contextual_priority_queue_lenses_node,
    stage4_context_complete_node,
)
from Stage5.nodes import stage5_diagnostics_node
from Stage6.nodes import prepare_stage6_node, stage6_complete_node, stage6_synthesis_node


def _add_common_nodes(workflow: StateGraph) -> None:
    workflow.add_node("load_data", load_data_node)
    workflow.add_node("stage1_prepare", prepare_stage1_node)
    workflow.add_node("attack_feasibility_critic", attack_feasibility_critic_node)
    workflow.add_node("defense_robustness_critic", defense_robustness_critic_node)
    workflow.add_node("stage1_complete", stage1_complete_node)
    workflow.add_node("stage2_prepare", prepare_stage2_node)
    workflow.add_node("stage2_debate", stage2_debate_node)
    workflow.add_node("stage2_complete", stage2_complete_node)
    workflow.add_node("stage3_build", stage3_build_node)

    workflow.set_entry_point("load_data")
    workflow.add_edge("load_data", "stage1_prepare")
    workflow.add_edge("stage1_prepare", "attack_feasibility_critic")
    workflow.add_edge("stage1_prepare", "defense_robustness_critic")
    workflow.add_edge(
        ["attack_feasibility_critic", "defense_robustness_critic"],
        "stage1_complete",
    )
    workflow.add_edge("stage1_complete", "stage2_prepare")
    workflow.add_edge("stage2_prepare", "stage2_debate")
    workflow.add_edge("stage2_debate", "stage2_complete")
    workflow.add_edge("stage2_complete", "stage3_build")


def create_graph():
    """Compile the active context-conditioned pipeline through constrained Stage 6."""
    workflow = StateGraph(AgentState)
    _add_common_nodes(workflow)
    workflow.add_node("stage3_contextualize", stage3_contextualize_node)
    workflow.add_node("stage4_context_prepare", prepare_stage4_context_node)
    workflow.add_node("contextual_priority_queue_lenses", contextual_priority_queue_lenses_node)
    workflow.add_node("stage4_context_complete", stage4_context_complete_node)
    workflow.add_node("stage5_diagnostics", stage5_diagnostics_node)
    workflow.add_node("stage6_prepare", prepare_stage6_node)
    workflow.add_node("stage6_synthesis", stage6_synthesis_node)
    workflow.add_node("stage6_complete", stage6_complete_node)

    workflow.add_edge("stage3_build", "stage3_contextualize")
    workflow.add_edge("stage3_contextualize", "stage4_context_prepare")
    workflow.add_edge("stage4_context_prepare", "contextual_priority_queue_lenses")
    workflow.add_edge("contextual_priority_queue_lenses", "stage4_context_complete")
    workflow.add_edge("stage4_context_complete", "stage5_diagnostics")
    workflow.add_edge("stage5_diagnostics", "stage6_prepare")
    workflow.add_edge("stage6_prepare", "stage6_synthesis")
    workflow.add_edge("stage6_synthesis", "stage6_complete")
    workflow.add_edge("stage6_complete", END)

    return workflow.compile()


def create_context_free_graph():
    """Reproduce the frozen Stage 4 v1 context-free baseline without changing its contract."""
    workflow = StateGraph(AgentState)
    _add_common_nodes(workflow)
    workflow.add_node("stage4_prepare", prepare_stage4_node)
    workflow.add_node("technical_feasibility_lens", technical_feasibility_lens_node)
    workflow.add_node("institutional_regional_lens", institutional_regional_lens_node)
    workflow.add_node("financial_adoption_lens", financial_adoption_lens_node)
    workflow.add_node("stage4_complete", stage4_complete_node)
    workflow.add_edge("stage3_build", "stage4_prepare")
    workflow.add_edge("stage4_prepare", "technical_feasibility_lens")
    workflow.add_edge("stage4_prepare", "institutional_regional_lens")
    workflow.add_edge("stage4_prepare", "financial_adoption_lens")
    workflow.add_edge(
        ["technical_feasibility_lens", "institutional_regional_lens", "financial_adoption_lens"],
        "stage4_complete",
    )
    workflow.add_edge("stage4_complete", END)
    return workflow.compile()
