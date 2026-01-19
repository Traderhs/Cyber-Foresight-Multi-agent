from langgraph.graph import StateGraph, END
from state import AgentState
from nodes import (
    attacker_node, 
    defender_node, 
    mediator_node, 
    load_data_node,
    technical_agent_node,
    regional_agent_node,
    finance_business_agent_node
)
MAX_ITERATIONS = 3  # global

def check_consensus(state: AgentState):
    """Analyze the last line of the "Mediator" review to determine the next path."""
    review = state.get("mediator_review", "").strip()
    count = state.get("iteration_count", 1)
    max_iterations = MAX_ITERATIONS

    # Check the last line is 'DECISION: DEBATE'
    is_debate = "DECISION: DEBATE" in review.upper().split('\n')[-1]

    if is_debate and count < max_iterations:
        print("-" * 50)
        print(f"Consensus Not Reached. (Turn: {count} / Remain Turn: {max_iterations-count})")
        return "attacker"

    print("\n" + "=" * 50)
    print("Consensus Reached or Max Iterations. \nMove to Technical Agent.")
    return "technical_agent"


def create_graph():
    workflow = StateGraph(AgentState)

    # Node Registration
    workflow.add_node("load_data", load_data_node)
    workflow.add_node("attacker", attacker_node)
    workflow.add_node("defender", defender_node)
    workflow.add_node("mediator", mediator_node)
    workflow.add_node("technical_agent", technical_agent_node)
    workflow.add_node("regional_agent", regional_agent_node)
    workflow.add_node("finance_business_agent", finance_business_agent_node)

    # Setting Entry Point
    workflow.set_entry_point("load_data")

    # Connect Edge
    workflow.add_edge("load_data", "attacker")
    workflow.add_edge("attacker", "defender")
    workflow.add_edge("defender", "mediator")

    # Setting Conditional Edge
    workflow.add_conditional_edges(
        "mediator",
        check_consensus,
        {
            "attacker": "attacker",
            "technical_agent": "technical_agent"
        }
    )

    # Sequential workflow for specialized agents
    workflow.add_edge("technical_agent", "regional_agent")
    workflow.add_edge("regional_agent", "finance_business_agent")
    workflow.add_edge("finance_business_agent", END)

    return workflow.compile()
