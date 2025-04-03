import json
import logging
import re # Add import for regex
from typing import List, Dict, Any, Optional

from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

# State definition
from .state import AgentState

# Modules for dependencies
from ..llms.provider import get_llm_instance
from ..rag.retriever import retrieve_rag_context, rag_initialized, initialization_error as rag_init_error
# Removed schema_loader import as schema comes from state now
from ..utils.mcp_utils import (
    prepare_mcp_sql_query_request,
    prepare_mcp_log_request,
    # Assume we'll need a function to call MCP tools, or handle it directly
    # Example: call_mcp_tool
)
# Import MCP components if needed for direct calls (adjust based on actual implementation)
# from .... # Need path to MCP SDK or wrapper if calling directly

logger = logging.getLogger(__name__)

# --- Node Functions ---

# Removed fetch_schema_node

def nlp_agent_node(state: AgentState) -> Dict[str, Any]:
    """
    Node to process the initial user query, refine it, and identify relevant tables.
    Placeholder implementation - currently just passes the query through.
    """
    thoughts = ["Executing NLP agent node..."]
    current_messages = state["messages"]
    user_query = current_messages[-1].content
    # Get schema directly from the state
    db_schema = state.get("db_schema")
    schema_error = state.get("schema_error") # Check if schema loading failed earlier

    if schema_error or not db_schema:
        thoughts.append(f"Skipping NLP analysis due to schema error from state: {schema_error or 'Schema not found in state.'}")
        # If schema failed or not in state, return immediately
        return {
            "refined_query": user_query,
            "relevant_tables": None,
            "agent_thoughts": state["agent_thoughts"] + thoughts,
            "error_message": schema_error
        }
    # --- Schema loaded successfully, continue ---
    thoughts.append(f"Using schema: {list(db_schema.keys())}")

    # --- Implement NLP Logic --- # This replaces the previous placeholder block
    refined_query = user_query # Default if refinement fails
    relevant_tables = None # Default
    nlp_error = None

    try:
        # 1. Get LLM instance
        # Use the same LLM selected by the user for consistency, or potentially a different one optimized for this task
        llm = get_llm_instance(state["selected_llm_name"])
        thoughts.append(f"Using LLM for NLP analysis: {state['selected_llm_name']}")

        # 2. Define Prompt for Table Extraction and Query Refinement
        nlp_prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are an analytical assistant. Your task is to analyze a user's database query and the database schema to identify relevant tables and refine the query if necessary for clarity.

            **Database Schema:**
            ```json
            {schema}
            ```

            **User Query:**
            "{query}"

            **Instructions:**
            1. Identify the database tables from the schema that are most likely needed to answer the user's query. Consider table names, column names, and the query's intent.
            2. If the user's query is ambiguous or could be phrased more clearly for SQL generation, provide a slightly refined version. If it's already clear, use the original query.
            3. Output your response ONLY as a JSON object with the following keys:
               - "relevant_tables": A list of strings containing the names of the relevant tables. If no specific tables seem relevant (e.g., query is too generic or unrelated), provide an empty list [].
               - "refined_query": A string containing the refined (or original) user query.

            Example Output: {{"relevant_tables": ["automobily", "ludia"], "refined_query": "Count the number of cars owned by people"}}

            IMPORTANT: Your entire response must be ONLY the JSON object, with no introductory text, explanations, or markdown formatting.
            """),
            # No history needed for this specific task usually
            # MessagesPlaceholder(variable_name="history"),
            # ("human", "{query}") # Query is already in the system prompt
        ])

        # 3. Prepare Chain
        # Using JsonOutputParser requires installing langchain_core[output_parsers] or similar
        # For simplicity, let's try StrOutputParser and parse JSON manually with error handling
        nlp_chain = nlp_prompt_template | llm | StrOutputParser()
        thoughts.append("Invoking LLM for NLP analysis...")

        # 4. Invoke Chain
        chain_input = {
            "schema": json.dumps(db_schema, indent=2),
            "query": user_query
        }
        llm_output_str = nlp_chain.invoke(chain_input)
        thoughts.append(f"NLP LLM Raw Output: {llm_output_str}")

        # 5. Parse JSON Output (More robust extraction)
        try:
            # Find the first '{' and the last '}'
            start_index = llm_output_str.find('{')
            end_index = llm_output_str.rfind('}')
            if start_index != -1 and end_index != -1 and end_index > start_index:
                json_str = llm_output_str[start_index:end_index+1].strip()
                thoughts.append(f"Extracted potential JSON: {json_str}")
                parsed_output = json.loads(json_str)
            else:
                raise ValueError("Could not find JSON object delimiters {{ ... }} in LLM output.")

            if isinstance(parsed_output, dict):
                 extracted_tables = parsed_output.get("relevant_tables")
                 extracted_query = parsed_output.get("refined_query")

                 # Basic validation
                 if isinstance(extracted_tables, list) and all(isinstance(t, str) for t in extracted_tables):
                      relevant_tables = extracted_tables
                 else:
                      thoughts.append("Warning: LLM output for 'relevant_tables' was not a list of strings.")
                      # Keep relevant_tables as None or [] ? Let's use None.

                 if isinstance(extracted_query, str) and extracted_query:
                      refined_query = extracted_query
                 else:
                      thoughts.append("Warning: LLM output for 'refined_query' was not a valid string. Using original.")
                      # Keep original user_query as refined_query

                 thoughts.append(f"NLP Analysis Result: Relevant Tables={relevant_tables}, Refined Query='{refined_query}'")

            else:
                 raise ValueError("LLM output was not a JSON dictionary.")

        except (json.JSONDecodeError, ValueError) as json_e:
            nlp_error = f"Failed to parse NLP LLM output as JSON: {json_e}. Raw: {llm_output_str}"
            logger.error(nlp_error)
            thoughts.append(nlp_error)
            # Keep defaults: refined_query = user_query, relevant_tables = None

    except Exception as e:
        nlp_error = f"Error during NLP analysis: {e}"
        logger.error(nlp_error, exc_info=True)
        thoughts.append(nlp_error)
        # Keep defaults on error

    # Update state, ensuring error message is passed if NLP failed critically
    # If only parsing failed, we might proceed with defaults but log the issue.
    # Let's pass nlp_error only if we want to halt execution based on it.
    # For now, we proceed with defaults if parsing fails.

    # Return results (or defaults if NLP failed)
    return {
        "refined_query": refined_query,
        "relevant_tables": relevant_tables,
        "agent_thoughts": state["agent_thoughts"] + thoughts,
        "error_message": nlp_error # Pass NLP error if one occurred
    }


def intent_classification_node(state: AgentState) -> Dict[str, Any]:
    """
    Classifies the user's query intent (SELECT, INSERT, UPDATE, DELETE, OTHER).
    """
    thoughts = ["Executing intent classification node..."]
    query_to_classify = state.get("refined_query") or state["messages"][-1].content
    intent = "UNSUPPORTED" # Default to unsupported
    intent_error = None

    # Simple keyword check for SELECT-like queries
    select_keywords = ["select", "show", "list", "count", "how many", "what is", "describe", "find"]
    query_lower = query_to_classify.lower()

    if any(keyword in query_lower for keyword in select_keywords):
         intent = "SELECT"
         thoughts.append("Classified intent as SELECT based on keywords.")
    else:
         # Check for DML keywords to explicitly classify as UNSUPPORTED
         dml_keywords = ["insert", "add", "update", "modify", "change", "delete", "remove", "drop", "create", "alter"]
         if any(keyword in query_lower for keyword in dml_keywords):
              intent = "UNSUPPORTED"
              thoughts.append("Classified intent as UNSUPPORTED (DML/DDL) based on keywords.")
         else:
              # If neither SELECT nor known DML/DDL, classify as OTHER (likely just a question)
              intent = "OTHER"
              thoughts.append("Classified intent as OTHER (non-query).")

    return {
        "query_intent": intent,
        "agent_thoughts": state["agent_thoughts"] + thoughts
        # "error_message": intent_error # Propagate error if needed
    }


def sql_generation_node(state: AgentState) -> Dict[str, Any]: # Node for SELECT
    """
    Node to generate the SQL query based on refined query, schema, RAG, and history.
    """
    thoughts = ["Executing SQL generation node..."]
    current_messages = state["messages"]
    # Use refined query if available, otherwise original user query
    query_to_use = state.get("refined_query") or current_messages[-1].content

    # 1. Get Schema from state
    db_schema = state.get("db_schema")
    schema_error = state.get("schema_error") # Check if schema loading failed earlier
    if schema_error or not db_schema:
        error_msg = f"Schema Error from state: {schema_error or 'Schema not found in state.'}"
        thoughts.append(error_msg)
        logger.error(error_msg)
        return {"agent_thoughts": state["agent_thoughts"] + thoughts, "error_message": error_msg}

    # 2. Check RAG status
    if not rag_initialized:
        thoughts.append(f"Warning: RAG not initialized ({rag_init_error}). Proceeding without RAG context.")
        rag_context = {"sql_examples": [], "gdrive_docs": []}
    else:
        rag_context = retrieve_rag_context(query_to_use)

    retrieved_sql_str = "\n".join([f"- {ex}" for ex in rag_context["sql_examples"]])
    retrieved_docs_str = "\n".join([f"- {doc}" for doc in rag_context["gdrive_docs"]])
    thoughts.append(f"Retrieved RAG context:\nSQL Examples:\n{retrieved_sql_str}\nGDrive Docs:\n{retrieved_docs_str}")

    # 3. Select LLM
    try:
        llm = get_llm_instance(state["selected_llm_name"])
        thoughts.append(f"Using LLM: {state['selected_llm_name']}")
    except ValueError as e:
        error_msg = f"LLM Selection Error: {e}"
        thoughts.append(error_msg)
        logger.error(error_msg)
        return {"agent_thoughts": state["agent_thoughts"] + thoughts, "error_message": error_msg}

    # 4. Define Prompt (Enhanced for more "agentic" behavior)
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are a highly skilled PostgreSQL AI Agent. Your primary function is to translate user requests into accurate, executable PostgreSQL queries.

        **Your Process:**
        1.  **Analyze the Request:** Carefully examine the user's query ({query}), the conversation history, and the provided database schema.
        2.  **Consider Context:** Use the retrieved SQL examples and document context for reference if they seem relevant, but prioritize the schema and the user's explicit request.
        3.  **Formulate the Query:** Construct a *single*, valid PostgreSQL query that directly addresses the user's request based on the available schema.
        4.  **Output ONLY SQL:** Your final output must be *only* the generated PostgreSQL query. Do not include any explanations, apologies, greetings, comments (outside of SQL syntax), or markdown formatting like ```sql.

        **Database Schema:**
        ```json
        {schema}
        ```

        **Retrieved SQL Examples (for reference only):**
        {sql_examples}

        **Retrieved Document Context (for reference only):**
        {doc_context}

        **Important Constraints:**
        - Output *only* the SQL query.
        - If the request is ambiguous, unclear, or cannot be answered with a single SQL query based on the schema, output the exact text: `Error: Cannot generate SQL for this query.`
        - Ensure the query is compatible with standard PostgreSQL syntax.
        """),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{query}") # User's current query
    ])

    # 5. Prepare Chain Input
    chain_input = {
        "schema": json.dumps(db_schema, indent=2), # Use loaded schema
        "sql_examples": retrieved_sql_str or "N/A",
        "doc_context": retrieved_docs_str or "N/A",
        "history": current_messages[:-1], # History excludes current query
        "query": query_to_use
    }

    # 6. Invoke LLM
    sql_generation_chain = prompt_template | llm | StrOutputParser()
    thoughts.append("Invoking LLM for SQL generation...")
    generated_sql: Optional[str] = None
    error_message: Optional[str] = None
    try:
        llm_output = sql_generation_chain.invoke(chain_input)
        thoughts.append(f"LLM Raw Output: {llm_output}")

        # Basic validation/cleanup
        cleaned_sql = llm_output.strip().replace("```sql", "").replace("```", "").strip()
        if not cleaned_sql or cleaned_sql.lower().startswith("error:") or ";" not in cleaned_sql:
            error_message = f"LLM failed to generate valid SQL. Response: {llm_output}"
            thoughts.append(error_message)
        else:
            generated_sql = cleaned_sql
            thoughts.append(f"Extracted SQL: {generated_sql}")

    except Exception as e:
        error_message = f"LLM Invocation Error: {e}"
        logger.error(error_message, exc_info=True)
        thoughts.append(error_message)

    return {
        "generated_sql": generated_sql,
        "agent_thoughts": state["agent_thoughts"] + thoughts,
        "error_message": error_message # Pass error message if generation failed
    }


# --- Removed DML Generation Node ---


def sql_validation_node(state: AgentState) -> Dict[str, Any]:
    """
    Node to validate the generated SQL query.
    Uses an LLM for checking common mistakes.
    """
    thoughts = ["Executing SQL validation node..."]
    generated_sql = state.get("generated_sql")
    error_message = state.get("error_message") # Check if generation already failed

    if error_message:
        thoughts.append(f"Skipping validation due to previous error: {error_message}")
        return {"agent_thoughts": state["agent_thoughts"] + thoughts, "validated_sql": None}

    if not generated_sql:
        thoughts.append("No SQL generated, skipping validation.")
        return {"agent_thoughts": state["agent_thoughts"] + thoughts, "validated_sql": None}

    # 1. Select LLM for checking
    try:
        # Maybe use a cheaper/faster model for validation? Or same one?
        llm = get_llm_instance(state["selected_llm_name"])
        thoughts.append(f"Using LLM for validation: {state['selected_llm_name']}")
    except ValueError as e:
        thoughts.append(f"LLM Selection Error for validation: {e}. Skipping validation.")
        logger.error(f"LLM Selection Error for validation: {e}")
        # Skip validation if checker LLM fails
        return {"agent_thoughts": state["agent_thoughts"] + thoughts, "validated_sql": generated_sql}

    # 2. Define check prompt
    query_check_system = """You are a PostgreSQL expert with a strong attention to detail. Double check the following PostgreSQL query for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers if needed (standard SQL quoting)
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

**Output Instructions:**
- If the query has mistakes, output *only* the corrected PostgreSQL query.
- If the query has no mistakes, output *only* the original PostgreSQL query.
- Do NOT include explanations, comments, markdown formatting (like ```sql), or any text other than the single, final SQL query."""

    query_check_prompt = ChatPromptTemplate.from_messages(
        [("system", query_check_system), ("human", "{query}")] # Pass only the query itself
    )
    checker_chain = query_check_prompt | llm | StrOutputParser()

    # 3. Invoke Checker LLM
    thoughts.append(f"Validating query: {generated_sql}")
    validated_sql = generated_sql # Default to original if check fails
    try:
        checked_sql = checker_chain.invoke({"query": generated_sql})
        thoughts.append(f"Validation LLM Raw Output: {checked_sql}")

        # --- Simplified Extraction Logic: Focus on last SQL block ---
        # --- Simplified Logic: Trust the strict prompt ---
        validated_sql = generated_sql # Default to original
        try:
            # Assume the LLM followed the strict output instructions
            llm_output_cleaned = checked_sql.strip()

            # Basic check: Does it look like SQL and is it non-empty?
            if llm_output_cleaned and ";" in llm_output_cleaned:
                thoughts.append(f"Validator LLM output (cleaned): {llm_output_cleaned}")
                # Use the output directly as the validated SQL
                validated_sql = llm_output_cleaned
                if llm_output_cleaned.lower() != generated_sql.lower():
                    thoughts.append("Using validator output as corrected SQL.")
                else:
                    thoughts.append("Validator output matches original SQL.")
            else:
                thoughts.append(f"Validator output doesn't look like valid SQL ('{llm_output_cleaned}'). Keeping original.")

        except Exception as parse_error: # Renamed 'e' to 'parse_error' to avoid conflict
            thoughts.append(f"Error processing validator output: {parse_error}. Keeping original query.")
            logger.warning(f"Error processing validator output: {parse_error}", exc_info=True)

    except Exception as e:
        logger.error(f"Error during SQL validation: {e}", exc_info=True)
        thoughts.append(f"SQL Validation Error: {e}. Keeping original query.")
        # Keep original SQL if checker fails

    return {
        "validated_sql": validated_sql,
        "agent_thoughts": state["agent_thoughts"] + thoughts
    }


def prepare_mcp_requests_node(state: AgentState) -> Dict[str, Any]:
    """
    Node to prepare the final MCP requests for query execution and logging.
    """
    thoughts = ["Executing MCP request preparation node..."]
    # Use the validated SQL if available, otherwise fallback (though likely None if validation skipped)
    sql_to_log_and_execute = state.get("validated_sql") or state.get("generated_sql")
    error_message = state.get("error_message") # Get potential error from generation/validation

    mcp_query_req = None
    mcp_log_req = None

    if sql_to_log_and_execute:
        thoughts.append(f"Preparing MCP requests for SQL: {sql_to_log_and_execute}")
        mcp_query_req = prepare_mcp_sql_query_request(sql_to_log_and_execute)
        mcp_log_req = prepare_mcp_log_request(sql_to_log_and_execute, error=error_message) # Log error if one occurred
        thoughts.append(f"Query Request Prepared: {mcp_query_req}")
        thoughts.append(f"Log Request Prepared: {mcp_log_req}")
    elif error_message:
         thoughts.append(f"SQL generation/validation failed: {error_message}. Skipping MCP request preparation.")
         # Optionally, prepare a log request just for the error?
         # mcp_log_req = prepare_mcp_log_request("N/A", error=error_message)
         # thoughts.append(f"Error Log Request Prepared: {mcp_log_req}")
    else:
         thoughts.append("No valid SQL available. Skipping MCP request preparation.")


    # Add final AI response to messages list
    final_response_content = ""
    query_intent = state.get("query_intent")

    if sql_to_log_and_execute:
        # Successfully generated SELECT query
        final_response_content = f"SQL query generated and validated. See execution status for MCP requests."
    elif error_message:
        # Error occurred during generation or validation
        final_response_content = f"Failed to generate SQL. Reason: {error_message}"
    elif query_intent == "UNSUPPORTED":
        # Intent was classified as unsupported DML/DDL
        final_response_content = "Sorry, I can only execute SELECT queries. Data modification requests (INSERT, UPDATE, DELETE) are not supported."
    elif query_intent == "OTHER":
         # Intent was classified as non-query
         final_response_content = "It seems like you're asking a question or making a statement that doesn't require a database query. How else can I help?"
    else:
        # Fallback for unexpected cases where no SQL and no specific error/intent
        final_response_content = "Agent finished without generating SQL or reporting an error."

    ai_message = AIMessage(content=final_response_content)

    return {
        "messages": state["messages"] + [ai_message], # Add final summary message
        "mcp_query_request": mcp_query_req,
        "mcp_log_request": mcp_log_req,
        "agent_thoughts": state["agent_thoughts"] + thoughts
    }
