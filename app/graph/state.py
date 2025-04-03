from typing import Annotated, List, Tuple, Optional, TypedDict, Dict, Any
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """
    Represents the state of our LangGraph agent.

    Attributes:
        messages: The list of messages exchanged between the user and the agent.
        selected_llm_name: The name of the LLM selected by the user.
        refined_query: The query potentially refined by the NLP agent node.
        relevant_tables: List of tables deemed relevant by the NLP agent node.
        generated_sql: The SQL query generated by the SQL generation node.
        validated_sql: The SQL query after validation/correction.
        mcp_query_request: The prepared MCP request dictionary for SQL execution.
        mcp_log_request: The prepared MCP request dictionary for SQL logging.
        agent_thoughts: A list accumulating thoughts/logs from each node.
        error_message: A potential error message to halt execution or display.
    """
    messages: Annotated[List[BaseMessage], add_messages]
    selected_llm_name: str

    # Fields populated by different nodes
    refined_query: Optional[str] = None
    relevant_tables: Optional[List[str]] = None
    generated_sql: Optional[str] = None
    validated_sql: Optional[str] = None
    mcp_query_request: Optional[Dict[str, Any]] = None
    mcp_log_request: Optional[Dict[str, Any]] = None
    agent_thoughts: List[str] # Use a list to append thoughts from each step
    error_message: Optional[str] = None
    query_intent: Optional[str] = None # Add field for query intent (SELECT, INSERT, UPDATE, DELETE, OTHER)
