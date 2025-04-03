import logging
from langgraph.graph import StateGraph, END, START

# Import state and node functions
from .state import AgentState
from .nodes import (
    nlp_agent_node,
    intent_classification_node, # Import new node
    sql_generation_node, # Will be for SELECT
    # dml_generation_node, # Removed DML node import
    sql_validation_node,
    prepare_mcp_requests_node
)
from typing import Literal

logger = logging.getLogger(__name__)

def build_graph() -> StateGraph:
    """
    Builds and compiles the LangGraph StateGraph for the SQL agent.
    """
    logger.info("Building LangGraph workflow...")
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("nlp_agent", nlp_agent_node)
    workflow.add_node("intent_classifier", intent_classification_node) # Add intent node
    workflow.add_node("sql_generator", sql_generation_node) # SELECT generator
    # workflow.add_node("dml_generator", dml_generation_node) # Removed DML node
    workflow.add_node("sql_validator", sql_validation_node)
    workflow.add_node("prepare_mcp_requests", prepare_mcp_requests_node)

    # Define Edges and Control Flow
    workflow.add_edge(START, "nlp_agent") # Start with NLP processing
    workflow.add_edge("nlp_agent", "intent_classifier") # After NLP, classify intent

    # Conditional Edges based on intent
    def decide_sql_generation_path(state: AgentState) -> Literal["sql_generator", "prepare_mcp_requests"]:
        """Determines the next node based on classified query intent."""
        intent = state.get("query_intent", "OTHER")
        logger.info(f"Routing based on intent: {intent}")
        if intent == "SELECT":
            return "sql_generator"
        # elif intent in ["INSERT", "UPDATE", "DELETE"]: # Route DML/UNSUPPORTED/OTHER to end for now
        #     return "dml_generator" # Removed DML path
        else: # UNSUPPORTED or OTHER intent
            # Skip SQL generation/validation if intent is not SELECT
            logger.warning(f"Query intent '{intent}' is not SELECT. Skipping SQL generation.")
            # We need to manually clear potential SQL fields if skipping generation?
            # This could be done here or by adding a dedicated 'skip_node'
            # Route directly to prepare_mcp_requests, which handles None SQL
            return "prepare_mcp_requests"

    workflow.add_conditional_edges(
        "intent_classifier",
        decide_sql_generation_path,
        {
            "sql_generator": "sql_generator",
            "prepare_mcp_requests": "prepare_mcp_requests" # Route UNSUPPORTED/OTHER directly to end
        }
    )

    # After SELECT SQL generation, proceed to validation
    workflow.add_edge("sql_generator", "sql_validator")

    # After validation, prepare the MCP requests
    workflow.add_edge("sql_validator", "prepare_mcp_requests")

    # After preparing requests, end the graph execution for this turn
    workflow.add_edge("prepare_mcp_requests", END)

    # Compile the graph
    graph = workflow.compile()
    logger.info("LangGraph workflow compiled successfully.")
    return graph

# Create a compiled instance for import elsewhere
compiled_graph = build_graph()
