import gradio as gr
import logging
import os
import json
from typing import List, Tuple, Dict, Any, Optional
import asyncio
import pandas as pd
from pydantic import SecretStr # Import SecretStr for masking

# --- Explicitly load and check settings first ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
try:
    from .config import settings
    # Log a non-secret setting to confirm loading
    logger.info(f"Settings loaded. Ollama model: {settings.ollama_model}")
except ImportError as ie:
     logger.critical(f"CRITICAL: Failed to import settings: {ie}", exc_info=True)
     # Exit or raise critical error if settings don't load
     raise SystemExit(f"Failed to load settings from config: {ie}")
except Exception as e:
     logger.critical(f"CRITICAL: Error accessing settings: {e}", exc_info=True)
     raise SystemExit(f"Error accessing settings: {e}")
# --- End explicit check ---


from langchain_core.messages import HumanMessage, AIMessage

# Import necessary components from modules
from .graph.state import AgentState
from .graph.builder import compiled_graph
from .rag.retriever import initialize_rag, rag_initialized, initialization_error as rag_init_error
from .rag.kb_manager import add_sql_examples_to_db, populate_gdrive_kb # Import populate_gdrive_kb
from .utils.mcp_utils import execute_mcp_tool, fetch_dynamic_schema, prepare_mcp_save_sql_request, prepare_mcp_read_file_request, prepare_mcp_log_request # Added prepare_mcp_log_request

# --- Global variables for initialized components ---
rag_success = False
rag_error = "Not initialized yet"
db_schema = None
schema_error = "Not initialized yet"

# --- Async Initialization Function ---
async def perform_initializations():
    """Initializes RAG and fetches dynamic schema."""
    global rag_success, rag_error, db_schema, schema_error
    logger.info("Performing initializations...")
    # Pass the imported settings object explicitly
    rag_success, rag_error = initialize_rag(settings)
    # Fetch schema dynamically
    db_schema, schema_error = await fetch_dynamic_schema()
    # Log results immediately after fetch
    logger.info(f"Initialization complete. RAG Success: {rag_success}, Schema Error: {schema_error}")
    logger.info(f"Global db_schema after fetch: {'Set' if db_schema is not None else 'None'}")
    logger.info(f"Global schema_error after fetch: {schema_error}")


# --- Gradio Interface Setup ---
# Get available LLMs from settings
available_llms = [llm.strip() for llm in settings.available_llms_str.split(',') if llm.strip()]
if not available_llms: # Fallback if setting is empty/invalid
    available_llms = ["Ollama (Local)", "OpenAI (API)", "Gemini (API)"]
    logger.warning("AVAILABLE_LLMS setting was empty or invalid, using default list.")

def get_initial_status() -> str:
    """Builds the initial status message based on component initialization."""
    status_parts = ["Status:"]
    # Schema Status
    if schema_error:
        status_parts.append(f"Schema Error ({schema_error})")
    elif db_schema:
        status_parts.append(f"Schema Loaded ({len(db_schema)} tables)") # Show count instead of keys
    else:
        status_parts.append("Schema Not Found")

    # RAG Status
    if not rag_success:
        status_parts.append(f"RAG Init Failed ({rag_error or 'Unknown Error'})")
    else:
        status_parts.append("RAG Initialized")

    return " | ".join(status_parts)

# --- Gradio Event Handlers ---

async def run_agent_graph_interface(
    user_query: str,
    selected_llm_name: str,
    chat_history: List[Tuple[str, str]]
) -> Tuple[List[Dict[str, str]], str, str, str, pd.DataFrame, Dict, Dict, str, gr.Button]:
    """
    Interface function for Gradio to invoke the LangGraph agent.
    Prepares SQL for approval but does NOT execute it.
    """
    logger.info(f"Gradio received query: '{user_query}' LLM: {selected_llm_name}")
    # Log global schema values at start of handler
    logger.info(f"run_agent_graph_interface: Start - Global db_schema is {'Set' if db_schema is not None else 'None'}, schema_error is {schema_error}")

    # 1. Convert Gradio history to Langchain messages
    messages = []
    for msg_data in chat_history:
        role = msg_data.get("role")
        content = msg_data.get("content", "")
        if role == "user": messages.append(HumanMessage(content=content))
        elif role == "assistant": messages.append(AIMessage(content=content))
        else: logger.warning(f"Unknown role in chat history: {role}")
    messages.append(HumanMessage(content=user_query))

    # 2. Prepare initial state
    initial_state: AgentState = {
        "messages": messages,
        "selected_llm_name": selected_llm_name,
        "agent_thoughts": [],
        "db_schema": db_schema, # Ensure global db_schema is passed
        "schema_error": schema_error, # Ensure global schema_error is passed
        "refined_query": None, "relevant_tables": None, "generated_sql": None,
        "validated_sql": None, "mcp_query_request": None, "mcp_log_request": None,
        "error_message": None
    }
    # Log state right after creation
    logger.info(f"run_agent_graph_interface: Initial state created - db_schema is {'Set' if initial_state.get('db_schema') is not None else 'None'}, schema_error is {initial_state.get('schema_error')}")


    # 3. Invoke graph
    try:
        final_state = compiled_graph.invoke(initial_state)
    except Exception as e:
        logger.error(f"Error invoking LangGraph: {e}", exc_info=True)
        final_state = {
            **initial_state,
            "messages": initial_state["messages"] + [AIMessage(content=f"Agent Error: {e}")],
            "agent_thoughts": initial_state["agent_thoughts"] + [f"GRAPH EXECUTION ERROR: {e}"],
            "error_message": str(e)
        }

    # 4. Extract results
    final_messages = final_state.get("messages", [])
    agent_sql_output = final_state.get("validated_sql") or final_state.get("generated_sql") or "-- No SQL Generated --"
    agent_thoughts_log = "\n".join(final_state.get("agent_thoughts", ["No thoughts recorded."]))
    error_message = final_state.get("error_message")
    mcp_query_req = final_state.get("mcp_query_request")
    mcp_log_req = final_state.get("mcp_log_request")

    # 5. Format status & prepare for approval
    exec_status = ""
    execute_button_update = gr.Button(interactive=False, visible=False) # Default hidden

    if agent_sql_output != "-- No SQL Generated --":
        exec_status += "SQL generated and validated.\n"
        if mcp_query_req:
             exec_status += f"\n**Prepared Query Request:**\n```json\n{json.dumps(mcp_query_req, indent=2)}\n```"
             exec_status += "\n\n**SQL ready for execution. Review and click 'Execute Approved SQL'.**"
             execute_button_update = gr.Button(interactive=True, visible=True) # Show & enable button
        else:
             # Should not happen if SQL is generated, but handle defensively
             exec_status += "\n\nError: SQL generated but no MCP query request prepared."
        if mcp_log_req:
             exec_status += f"\n\n**Prepared Log Request (will execute on approval):**\n```json\n{json.dumps(mcp_log_req, indent=2)}\n```"
    elif error_message:
        exec_status = f"SQL generation/validation failed.\nReason: {error_message}"
    else:
        # Handle cases where no SQL was generated (e.g., unsupported intent)
        last_msg_content = final_messages[-1].content if final_messages else "Agent finished."
        exec_status = f"Agent finished. Reason: {last_msg_content}"

    # 6. Convert messages back to Gradio format
    new_chat_history_messages = []
    for msg in final_messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        content = getattr(msg, 'content', str(msg))
        new_chat_history_messages.append({"role": role, "content": content})

    # Ensure list isn't empty if error occurred early
    if not new_chat_history_messages and messages:
         new_chat_history_messages.append({"role": "user", "content": user_query})
         new_chat_history_messages.append({"role": "assistant", "content": f"Error: {error_message or 'Unknown agent error'}"})

    # Return updates for UI components AND state components
    return (
        new_chat_history_messages,
        agent_sql_output,
        agent_thoughts_log,
        exec_status,
        pd.DataFrame(), # Clear previous results
        mcp_query_req or {}, # Store request in state
        mcp_log_req or {}, # Store log request in state
        agent_sql_output if mcp_query_req else "", # Store SQL in state only if query req exists
        execute_button_update # Update execute button visibility/interactivity
    )

# --- Human-in-the-Loop Execution Handler ---
async def execute_approved_sql_handler(
    mcp_query_req_state: Dict,
    mcp_log_req_state: Dict,
    agent_sql_output_state: str # The actual SQL query string
) -> Tuple[pd.DataFrame, str, gr.Button]:
    """
    Executes the SQL query stored in state after user clicks the approval button.
    Also handles logging and saving the SQL upon successful execution.
    """
    exec_status = ""
    mcp_result_display = pd.DataFrame()
    # Hide button immediately after click while processing
    execute_button_update = gr.Button(interactive=False, visible=False)

    if not mcp_query_req_state:
        exec_status = "**Error: No query request found in state to execute.**"
        logger.error("Execute button clicked but no query request in state.")
        return mcp_result_display, exec_status, execute_button_update

    logger.info(f"Executing approved MCP query request: {mcp_query_req_state}")
    exec_status = f"Executing approved SQL...\nRequest:\n```json\n{json.dumps(mcp_query_req_state, indent=2)}\n```\n"

    query_success = False # Flag to track if query execution succeeded

    try:
        # --- Execute Query ---
        mcp_response = await execute_mcp_tool(mcp_query_req_state)
        logger.info(f"Approved MCP Query Response: {mcp_response}")

        if mcp_response.get("success"):
            query_success = True # Mark query as successful
            exec_status += "\n**Query Execution Successful.**"
            raw_result = mcp_response.get("result")
            # Attempt to convert to DataFrame
            if isinstance(raw_result, list) and raw_result and isinstance(raw_result[0], dict):
                try:
                    mcp_result_display = pd.DataFrame(raw_result)
                    logger.info("Converted MCP result to DataFrame.")
                except Exception as df_e:
                    logger.error(f"Failed to convert MCP result to DataFrame: {df_e}")
                    mcp_result_display = pd.DataFrame({'Error': [f"Failed to display results: {df_e}"]})
                    exec_status += f"\n**Result Display Error:** Failed to convert results to table: {df_e}"
            elif isinstance(raw_result, str):
                 mcp_result_display = pd.DataFrame({'Message': [raw_result]})
                 logger.info(f"MCP result is a string message: {raw_result}")
            elif isinstance(raw_result, dict):
                 try:
                      mcp_result_display = pd.DataFrame([raw_result])
                      logger.info("Converted single-row MCP result dict to DataFrame.")
                 except Exception as df_e:
                      logger.error(f"Failed to convert MCP result dict to DataFrame: {df_e}")
                      mcp_result_display = pd.DataFrame({'Error': [f"Failed to display dict result: {df_e}"]})
                      exec_status += f"\n**Result Display Error:** Failed to convert dict result to table: {df_e}"
            else:
                 exec_status += "\nQuery successful, but no displayable content returned."

        else: # MCP query call failed
            error_msg = mcp_response.get('error', 'MCP query execution failed.')
            mcp_result_display = pd.DataFrame({'Error': [error_msg]})
            exec_status += f"\n**Query Execution Failed:** {error_msg}"
            logger.error(f"Approved MCP query failed: {error_msg}")

    except Exception as mcp_e:
        logger.error(f"Error calling execute_mcp_tool for approved query: {mcp_e}", exc_info=True)
        mcp_result_display = pd.DataFrame({'Error': [f"Failed to execute approved query via MCP: {mcp_e}"]})
        exec_status += f"\n**Query Execution Exception:** {mcp_e}"

    # --- Execute Log and Save SQL only if Query Succeeded ---
    if query_success:
        # --- Execute Log ---
        if mcp_log_req_state:
            logger.info(f"Attempting to execute MCP log request: {mcp_log_req_state}")
            try:
                # Prepare log request again, potentially with updated info if needed
                # For now, assume the state one is fine
                log_response = await execute_mcp_tool(mcp_log_req_state)
                logger.info(f"MCP Log Response: {log_response}")
                if not log_response.get("success"):
                     logger.error(f"MCP logging failed: {log_response.get('error', 'Unknown error')}")
                     exec_status += f"\n**MCP Logging Failed:** {log_response.get('error', 'Unknown error')}"
                else:
                     logger.info("MCP logging successful.")
                     exec_status += "\n**MCP Logging Successful.**"
            except Exception as log_e:
                logger.error(f"Error calling execute_mcp_tool for logging: {log_e}", exc_info=True)
                exec_status += f"\n**MCP Logging Exception:** {log_e}"

        # --- Save SQL ---
        if agent_sql_output_state and agent_sql_output_state != "-- No SQL Generated --":
             try:
                 save_sql_req = prepare_mcp_save_sql_request(agent_sql_output_state)
                 logger.info(f"Attempting to save successful SQL: {save_sql_req}")
                 save_response = await execute_mcp_tool(save_sql_req)
                 if not save_response.get("success"):
                     logger.error(f"Failed to save successful SQL: {save_response.get('error')}")
                     exec_status += f"\n**Failed to save SQL:** {save_response.get('error')}"
                 else:
                     logger.info("Successfully saved SQL query.")
                     exec_status += "\n**SQL query saved.**"
             except Exception as save_e:
                 logger.error(f"Exception while saving SQL: {save_e}", exc_info=True)
                 exec_status += f"\n**Exception saving SQL:** {save_e}"
    else:
        exec_status += "\n\nSkipping logging and saving due to query execution failure."


    return mcp_result_display, exec_status, execute_button_update


# --- KB Update Handler ---
async def update_sql_kb_handler() -> str:
    """
    Handles the button click to update the SQL KB from the saved queries file.
    """
    # Use path from settings
    saved_sql_path = settings.mcp_saved_sql_path
    logger.info(f"Initiating SQL KB update from {saved_sql_path}...")

    # 1. Prepare MCP request to read the file
    read_request = prepare_mcp_read_file_request(saved_sql_path)

    # 2. Execute MCP request
    try:
        read_response = await execute_mcp_tool(read_request)
        logger.info(f"MCP Read File Response: {read_response}")

        if not read_response.get("success"):
            error_msg = read_response.get('error', 'Unknown MCP error')
            # Check if the error indicates file not found (adjust pattern as needed based on server)
            # More robust check for file not found errors
            if isinstance(error_msg, str) and ("not found" in error_msg.lower() or "no such file" in error_msg.lower()):
                 logger.warning(f"Saved SQL file {saved_sql_path} not found via MCP.")
                 return "Saved SQL file not found. No queries added to KB yet."
            else:
                 # Handle other MCP errors
                 logger.error(f"Failed to read {saved_sql_path} via MCP: {error_msg}")
                 return f"Error reading saved SQL file: {error_msg}"

        # Content should be in result if successful (check mcp_utils parsing)
        file_content = read_response.get("result")
        if not isinstance(file_content, str):
             error_msg = f"Unexpected content type received from MCP read_file: {type(file_content)}. Expected string."
             logger.error(error_msg)
             file_content = str(file_content) if file_content is not None else ""
             if not file_content:
                  return f"Error: {error_msg}"

        if not file_content.strip():
             logger.info(f"{saved_sql_path} is empty or contains only whitespace.")
             return "No new SQL queries found in the saved file to add to KB."

        # 3. Parse SQL queries
        sql_queries = [q.strip() for q in file_content.split(';') if q.strip()]
        logger.info(f"Parsed {len(sql_queries)} potential SQL queries from file.")

        if not sql_queries:
            return "Parsed file content but found no valid SQL queries to add."

        # 4. Add queries to ChromaDB
        added_count, status_msg = add_sql_examples_to_db(sql_queries)

        return status_msg # Return the status from the add function

    except Exception as e: # Catch other potential exceptions during processing
        error_msg = f"Error during SQL KB update: {e}"
        logger.error(error_msg, exc_info=True)
        return f"Error: {error_msg}"


# --- Build Gradio UI ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    # --- State Components ---
    mcp_query_req_state = gr.State({})
    mcp_log_req_state = gr.State({})
    agent_sql_output_state = gr.State("")
    # --- End State Components ---

    gr.Markdown("# PostgreSQL AI Agent (LangGraph + MCP Mode)")
    status_display = gr.Textbox(label="Status", value=get_initial_status(), interactive=False)

    # Display Loaded Configuration
    with gr.Accordion("Loaded Configuration", open=False):
        config_display_dict = {}
        for key, value in settings.model_dump().items():
            if hasattr(value, 'get_secret_value') and callable(value.get_secret_value):
                 config_display_dict[key] = "****"
            else:
                 config_display_dict[key] = value
        gr.JSON(value=config_display_dict, label="Current Settings (from .env & defaults)")

    with gr.Row():
        # Column for Controls
        with gr.Column(scale=1):
            llm_selector = gr.Dropdown(label="Select LLM", choices=available_llms, value=available_llms[0])
            query_input = gr.Textbox(label="Your Query", placeholder="e.g., Show me all users from the 'customers' table", lines=4)
            submit_button = gr.Button("Generate SQL", variant="primary")
            # Add Execute Button (initially hidden)
            execute_sql_button = gr.Button("Execute Approved SQL", variant="stop", visible=False, interactive=False)
        # Column for Chatbot
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Conversation", height=500, type="messages")

    # Accordion for Details below the main interaction
    with gr.Accordion("Agent Details & Generated SQL", open=False):
         with gr.Row():
             agent_steps_output = gr.Markdown(label="Agent Execution Log")
         with gr.Row():
             with gr.Column(scale=2):
                 sql_output = gr.Code(label="Generated/Validated SQL", language="sql", interactive=False)
             with gr.Column(scale=1):
                 execution_status_output = gr.Textbox(label="Execution Status & MCP Requests", lines=10, interactive=False, show_copy_button=True)
         with gr.Row():
             query_results_output = gr.DataFrame(label="Query Results", wrap=True)

    # KB Update Controls
    with gr.Row():
         update_sql_kb_button = gr.Button("Update SQL KB from Saved Queries")
         gdrive_kb_button = gr.Button("Prepare GDrive KB Population (Manual Steps)") # New button
         kb_status = gr.Textbox(label="KB Update Status", interactive=False)

    # --- Event Handlers ---
    submit_button.click(
        fn=run_agent_graph_interface,
        inputs=[query_input, llm_selector, chatbot],
        # Outputs now include state variables and the execute button
        outputs=[
            chatbot, sql_output, agent_steps_output, execution_status_output,
            query_results_output, # Clear results
            mcp_query_req_state, mcp_log_req_state, agent_sql_output_state, # Update state
            execute_sql_button # Update button visibility
        ],
        api_name="run_agent_graph"
    )

    # Handler for the execute button
    execute_sql_button.click(
        fn=execute_approved_sql_handler,
        inputs=[mcp_query_req_state, mcp_log_req_state, agent_sql_output_state],
        outputs=[query_results_output, execution_status_output, execute_sql_button], # Update results, status, and hide button
        api_name="execute_approved_sql"
    )

    update_sql_kb_button.click(fn=update_sql_kb_handler, inputs=[], outputs=[kb_status], api_name="update_sql_kb")
    # Connect the GDrive button to the placeholder function which now returns instructions
    gdrive_kb_button.click(fn=populate_gdrive_kb, inputs=[], outputs=[kb_status], api_name="populate_gdrive_kb")
    submit_button.click(lambda: "", inputs=[], outputs=[query_input]) # Clear input

# --- Launch App ---
async def main():
    # Perform initializations asynchronously
    await perform_initializations()

    # Build the Gradio UI (must be done after initializations)
    # UI is now defined above, just launch it here
    logger.info("Starting Gradio application...")
    try:
        new_port = 7864
        logger.info(f"Attempting to launch Gradio on port {new_port}...")
        demo.launch(server_name="0.0.0.0", server_port=new_port, share=False)
    except OSError as e:
        logger.error(f"Gradio launch failed on port {new_port}: {e}", exc_info=True)
    except Exception as launch_e:
         logger.error(f"Gradio launch failed: {launch_e}", exc_info=True)
    logger.info("Gradio application stopped.")


if __name__ == "__main__":
    # Run the main async function
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application interrupted by user.")
    except Exception as e:
        logger.critical(f"Application failed to run: {e}", exc_info=True)
