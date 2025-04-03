import gradio as gr
import gradio as gr
import logging
import os
import json
from typing import List, Tuple, Dict, Any, Optional # Add Optional
import asyncio # Add asyncio
import pandas as pd # Re-add pandas import

# --- Explicitly load and check settings first ---
import logging # Need logging early
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Initialize logger early
try:
    from .config import settings
    loaded_key = settings.openai_api_key
    logger.info(f"EARLY DEBUG in main.py: Loaded openai_api_key type: {type(loaded_key)}")
    if loaded_key:
        logger.info(f"EARLY DEBUG in main.py: Loaded openai_api_key value (partial): {loaded_key.get_secret_value()[:5]}...{loaded_key.get_secret_value()[-4:]}")
    else:
        logger.info("EARLY DEBUG in main.py: Loaded openai_api_key is None.")
except ImportError as ie:
     logger.error(f"EARLY DEBUG in main.py: Failed to import settings: {ie}")
except Exception as e:
     logger.error(f"EARLY DEBUG in main.py: Error accessing settings: {e}")
# --- End explicit check ---


from langchain_core.messages import HumanMessage, AIMessage

# Import necessary components from modules
from .config import settings
from .graph.state import AgentState
from .graph.builder import compiled_graph # Import the compiled graph
from .rag.retriever import initialize_rag, rag_initialized, initialization_error as rag_init_error
from .rag.kb_manager import populate_gdrive_kb # Keep the placeholder KB manager function
# from .utils.schema_loader import load_schema_from_file, get_schema # Removed static schema loader imports
from .utils.mcp_utils import execute_mcp_tool, fetch_dynamic_schema # Remove placeholder import

# Configure logging (already done above)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

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
    db_schema, schema_error = await fetch_dynamic_schema() # Call the new async function
    logger.info(f"Initialization complete. RAG Success: {rag_success}, Schema Error: {schema_error}")

# --- Gradio Interface Setup ---
available_llms = ["Ollama (Local)", "OpenAI (API)", "Gemini (API)"]

def get_initial_status() -> str:
    """Builds the initial status message based on component initialization."""
    status_parts = ["Status:"]
    # Schema Status
    if schema_error:
        status_parts.append(f"Schema Error ({schema_error})")
    elif db_schema:
        status_parts.append(f"Schema Loaded ({list(db_schema.keys())})")
    else:
        status_parts.append("Schema Not Found") # Should be covered by schema_error

    # RAG Status
    if not rag_success:
        status_parts.append(f"RAG Init Failed ({rag_error or 'Unknown Error'})")
    else:
        status_parts.append("RAG Initialized")

    return " | ".join(status_parts)

# --- Gradio Event Handler ---
# Make the function async again to allow await for MCP call
async def run_agent_graph_interface(
    user_query: str,
    selected_llm_name: str,
    chat_history: List[Tuple[str, str]] # Input is still list of tuples from Gradio state
) -> Tuple[List[Dict[str, str]], str, str, str, Optional[Any]]: # Restore result to return type
    """
    Interface function for Gradio to invoke the LangGraph agent.
    Handles input/output conversion between Gradio and LangGraph state.
    """
    logger.info(f"Gradio received query: '{user_query}' LLM: {selected_llm_name}")

    # 1. Convert Gradio 'messages' history (List[Dict]) to Langchain messages
    messages = []
    # The input 'chat_history' from Gradio (type='messages') is List[Dict[str, str]]
    for msg_data in chat_history:
        role = msg_data.get("role")
        content = msg_data.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
        else:
            logger.warning(f"Unknown role in chat history: {role}")
            # Handle unknown roles if necessary, maybe append as system message?
            # messages.append(SystemMessage(content=f"Unknown role '{role}': {content}"))

    # Add the *new* user query from the input box
    messages.append(HumanMessage(content=user_query))

    # 2. Prepare initial state for the graph
    # Ensure agent_thoughts is initialized as a list
    initial_state: AgentState = {
        "messages": messages,
        "selected_llm_name": selected_llm_name,
        "agent_thoughts": [], # Initialize as empty list
        "db_schema": db_schema, # Re-add schema to initial state
        # Other state fields will be populated by the graph nodes
        "refined_query": None,
        "relevant_tables": None,
        "generated_sql": None,
        "validated_sql": None,
        "mcp_query_request": None,
        "mcp_log_request": None,
        "error_message": None
    }

    # 3. Invoke the compiled graph
    try:
        # Use the pre-compiled graph instance
        # Remove config argument
        final_state = compiled_graph.invoke(initial_state)
    except Exception as e:
        logger.error(f"Error invoking LangGraph: {e}", exc_info=True)
        # Handle graph execution error gracefully
        final_state = {
            **initial_state,
            "messages": initial_state["messages"] + [AIMessage(content=f"Agent Error: {e}")],
            "agent_thoughts": initial_state["agent_thoughts"] + [f"GRAPH EXECUTION ERROR: {e}"],
            "error_message": str(e)
        }


    # 4. Extract results from the final state
    final_messages = final_state.get("messages", [])
    # Use validated_sql if available, otherwise generated_sql
    agent_sql_output = final_state.get("validated_sql") or final_state.get("generated_sql") or "-- No SQL Generated --"
    agent_thoughts_log = "\n".join(final_state.get("agent_thoughts", ["No thoughts recorded."]))
    error_message = final_state.get("error_message")

    # 5. Format execution status including MCP requests
    mcp_query_req = final_state.get("mcp_query_request")
    mcp_log_req = final_state.get("mcp_log_request")
    exec_status = ""

    if agent_sql_output != "-- No SQL Generated --":
        exec_status += "SQL generated and validated.\n"
        if mcp_query_req:
             # Use json.dumps for proper formatting
             exec_status += f"\n**Execute via MCP Client:**\n```json\n{json.dumps(mcp_query_req, indent=2)}\n```"
        if mcp_log_req:
             # Warning about overwrite
             exec_status += f"\n\n**Log via MCP Client (Overwrites!):**\n```json\n{json.dumps(mcp_log_req, indent=2)}\n```"
    elif error_message:
        exec_status = f"SQL generation/validation failed.\nReason: {error_message}"
    else:
        # Check the last message for potential errors if state doesn't have specific error
        if final_messages and isinstance(final_messages[-1], AIMessage):
             ai_content = final_messages[-1].content
             if ai_content.lower().startswith("error:"):
                  exec_status = f"SQL Generation Failed.\nReason: {ai_content}"
             else:
                  exec_status = "Agent finished without generating SQL."
        else:
             exec_status = "Agent finished without generating SQL or reporting an error."

    # --- Execute Real MCP Query ---
    mcp_result_display = pd.DataFrame() # Default to empty DataFrame
    if mcp_query_req:
        logger.info(f"Attempting to execute real MCP query request: {mcp_query_req}")
        try:
            # Call the actual MCP execution function from mcp_utils
            mcp_response = await execute_mcp_tool(mcp_query_req)
            logger.info(f"Real MCP Response: {mcp_response}") # Log the actual response

            if mcp_response.get("success"):
                raw_result = mcp_response.get("result")
                # Attempt to convert to DataFrame ONLY if it's a list of dicts
                if isinstance(raw_result, list) and raw_result and isinstance(raw_result[0], dict):
                    try:
                        mcp_result_display = pd.DataFrame(raw_result)
                        logger.info("Converted MCP result to DataFrame.")
                    except Exception as df_e:
                        logger.error(f"Failed to convert MCP result to DataFrame: {df_e}")
                        # Display error in a DataFrame format
                        mcp_result_display = pd.DataFrame({'Error': [f"Failed to display results: {df_e}"]})
                elif isinstance(raw_result, str):
                     # If it's just a string message, display that
                     mcp_result_display = pd.DataFrame({'Message': [raw_result]})
                     logger.info(f"MCP result is a string message: {raw_result}")
                elif isinstance(raw_result, dict):
                     # Handle dictionary results (e.g., single row or status)
                     try:
                          mcp_result_display = pd.DataFrame([raw_result]) # Wrap dict in a list
                          logger.info("Converted single-row MCP result dict to DataFrame.")
                     except Exception as df_e:
                          logger.error(f"Failed to convert MCP result dict to DataFrame: {df_e}")
                          mcp_result_display = pd.DataFrame({'Error': [f"Failed to display dict result: {df_e}"]})
                # else: # If raw_result is None or empty list, mcp_result_display remains empty DataFrame

            else: # MCP call failed
                error_msg = mcp_response.get('error', 'MCP call failed.')
                mcp_result_display = pd.DataFrame({'Error': [error_msg]})
                logger.error(f"MCP call failed: {error_msg}")
        except Exception as mcp_e:
            logger.error(f"Error calling execute_mcp_tool: {mcp_e}", exc_info=True)
            mcp_result_display = pd.DataFrame({'Error': [f"Failed to execute query via MCP: {mcp_e}"]})
    else:
        logger.info("No MCP query request to execute.")
        # mcp_result_display remains empty DataFrame

    # --- Execute Real MCP Log ---
    if mcp_log_req:
        logger.info(f"Attempting to execute real MCP log request: {mcp_log_req}")
        try:
            log_response = await execute_mcp_tool(mcp_log_req) # execute_mcp_tool now handles filesystem
            logger.info(f"Real MCP Log Response: {log_response}")
            if not log_response.get("success"):
                 logger.error(f"MCP logging failed: {log_response.get('error', 'Unknown error')}")
                 # Optionally update status or display error? Maybe add to exec_status?
                 exec_status += f"\n\n**MCP Logging Failed:** {log_response.get('error', 'Unknown error')}"
            else:
                 logger.info("MCP logging successful.")
                 # Optionally add success message to exec_status?
                 # exec_status += "\n\n**MCP Logging Successful.**"
        except Exception as log_e:
            logger.error(f"Error calling execute_mcp_tool for logging: {log_e}", exc_info=True)
            # Optionally update status or display error?
            exec_status += f"\n\n**MCP Logging Exception:** {log_e}"


    # 6. Convert Langchain messages back to Gradio 'messages' format
    # Expected format: List[Dict[str, str]] with keys 'role' ('user' or 'assistant') and 'content'
    new_chat_history_messages = []
    for msg in final_messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        # Handle potential errors or unexpected message types gracefully
        content = getattr(msg, 'content', str(msg))
        new_chat_history_messages.append({"role": role, "content": content})

    # Ensure the list isn't empty if there was an error early on
    if not new_chat_history_messages and messages:
         # Add the original user query and the error message
         new_chat_history_messages.append({"role": "user", "content": user_query})
         new_chat_history_messages.append({"role": "assistant", "content": f"Error: {error_message or 'Unknown agent error'}"})


    # Return the display-formatted MCP result and the new message format
    return new_chat_history_messages, agent_sql_output, agent_thoughts_log, exec_status, mcp_result_display


# --- Build Gradio UI ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# PostgreSQL AI Agent (LangGraph + MCP Mode)")
    # Use a status Textbox that can be updated
    status_display = gr.Textbox(label="Status", value=get_initial_status(), interactive=False)

    with gr.Row():
        # Column for Controls
        with gr.Column(scale=1):
            llm_selector = gr.Dropdown(label="Select LLM", choices=available_llms, value=available_llms[0])
            query_input = gr.Textbox(label="Your Query", placeholder="e.g., Show me all users from the 'customers' table", lines=4)
            submit_button = gr.Button("Generate SQL", variant="primary")
        # Column for Chatbot
        with gr.Column(scale=2):
            # Update chatbot parameters to address warnings
            chatbot = gr.Chatbot(label="Conversation", height=500, type="messages") # Use type='messages'

    # Accordion for Details below the main interaction
    with gr.Accordion("Agent Details & Generated SQL", open=False): # Start closed by default
         with gr.Row():
             # Use Markdown for Agent Steps for better formatting potential
             agent_steps_output = gr.Markdown(label="Agent Execution Log")
         with gr.Row():
             # SQL Output and Execution Status side-by-side
             with gr.Column(scale=2):
                 sql_output = gr.Code(label="Generated/Validated SQL", language="sql", interactive=False)
             with gr.Column(scale=1):
                 execution_status_output = gr.Textbox(label="Execution Status & MCP Requests", lines=10, interactive=False, show_copy_button=True)
         # Restore row for Query Results display
         with gr.Row():
             query_results_output = gr.DataFrame(label="Query Results", wrap=True)

    # Keep KB controls at the bottom
    with gr.Row():
         kb_button = gr.Button("Populate KB from GDrive (Placeholder)")
         kb_status = gr.Textbox(label="KB Status", interactive=False)

    # --- Event Handlers ---
    submit_button.click(
        fn=run_agent_graph_interface, # Use the interface function
        inputs=[query_input, llm_selector, chatbot],
        # Restore query_results_output to the outputs list
        outputs=[chatbot, sql_output, agent_steps_output, execution_status_output, query_results_output],
        api_name="run_agent_graph"
    )
    kb_button.click(fn=populate_gdrive_kb, inputs=[], outputs=[kb_status], api_name="populate_gdrive_kb")
    submit_button.click(lambda: "", inputs=[], outputs=[query_input]) # Clear input

# --- Launch App ---
async def main():
    # Perform initializations asynchronously
    await perform_initializations()

    # Build the Gradio UI (must be done after initializations)
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# PostgreSQL AI Agent (LangGraph + MCP Mode)")
        # Use a status Textbox that can be updated
        status_display = gr.Textbox(label="Status", value=get_initial_status(), interactive=False)

        with gr.Row():
            # Column for Controls
            with gr.Column(scale=1):
                llm_selector = gr.Dropdown(label="Select LLM", choices=available_llms, value=available_llms[0])
                query_input = gr.Textbox(label="Your Query", placeholder="e.g., Show me all users from the 'customers' table", lines=4)
                submit_button = gr.Button("Generate SQL", variant="primary")
            # Column for Chatbot
            with gr.Column(scale=2):
                # Update chatbot parameters to address warnings
                chatbot = gr.Chatbot(label="Conversation", height=500, type="messages") # Use type='messages'

        # Accordion for Details below the main interaction
        with gr.Accordion("Agent Details & Generated SQL", open=False): # Start closed by default
             with gr.Row():
                 # Use Markdown for Agent Steps for better formatting potential
                 agent_steps_output = gr.Markdown(label="Agent Execution Log")
             with gr.Row():
                 # SQL Output and Execution Status side-by-side
                 with gr.Column(scale=2):
                     sql_output = gr.Code(label="Generated/Validated SQL", language="sql", interactive=False)
                 with gr.Column(scale=1):
                     execution_status_output = gr.Textbox(label="Execution Status & MCP Requests", lines=10, interactive=False, show_copy_button=True)
             # Restore row for Query Results display
             with gr.Row():
                 query_results_output = gr.DataFrame(label="Query Results", wrap=True)

        # Keep KB controls at the bottom
        with gr.Row():
             kb_button = gr.Button("Populate KB from GDrive (Placeholder)")
             kb_status = gr.Textbox(label="KB Status", interactive=False)

        # --- Event Handlers ---
        submit_button.click(
            fn=run_agent_graph_interface, # Use the interface function
            inputs=[query_input, llm_selector, chatbot],
            # Restore query_results_output to the outputs list
            outputs=[chatbot, sql_output, agent_steps_output, execution_status_output, query_results_output],
            api_name="run_agent_graph"
        )
        kb_button.click(fn=populate_gdrive_kb, inputs=[], outputs=[kb_status], api_name="populate_gdrive_kb")
        submit_button.click(lambda: "", inputs=[], outputs=[query_input]) # Clear input

    # Launch the Gradio app
    logger.info("Starting Gradio application...")
    try:
        # Try yet another port
        new_port = 7864
        logger.info(f"Attempting to launch Gradio on port {new_port}...")
        # Need to launch in a way that allows the async event loop to run
        # demo.launch() is blocking, so run initializations before this point.
        # The Gradio event handler `run_agent_graph_interface` is already async.
        demo.launch(server_name="0.0.0.0", server_port=new_port, share=False)
    except OSError as e:
        # Handle specific port-in-use error if needed, or just log general error
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
