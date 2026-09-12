from langgraph.graph import StateGraph, END
from Pipeline.state import AgentState
from Stage0.nodes import load_data_node
from Stage1.nodes import (
    prepare_stage1_node,
    attack_feasibility_critic_node,
    defense_robustness_critic_node,
    stage1_complete_node,
)


def create_graph():
    """Compile the implemented pipeline through Stage 1 only.

    Stage 2+ legacy code remains in the repository but is deliberately not wired
    into the active graph until its redesigned contract is implemented.
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("load_data", load_data_node)
    workflow.add_node("stage1_prepare", prepare_stage1_node)
    workflow.add_node("attack_feasibility_critic", attack_feasibility_critic_node)
    workflow.add_node("defense_robustness_critic", defense_robustness_critic_node)
    workflow.add_node("stage1_complete", stage1_complete_node)

    workflow.set_entry_point("load_data")

    workflow.add_edge("load_data", "stage1_prepare")

    # Both critics receive the same Stage 0 state and exact Stage 1 artifact
    # identity in the same superstep. Neither
    # branch can observe the other's initial assessment before the join.
    workflow.add_edge("stage1_prepare", "attack_feasibility_critic")
    workflow.add_edge("stage1_prepare", "defense_robustness_critic")
    workflow.add_edge(
        ["attack_feasibility_critic", "defense_robustness_critic"],
        "stage1_complete",
    )
    workflow.add_edge("stage1_complete", END)

    return workflow.compile()
