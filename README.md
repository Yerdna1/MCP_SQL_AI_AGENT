# PostgreSQL AI Agent (LangGraph + MCP Mode)

This project implements an AI agent capable of interacting with a PostgreSQL database based on natural language queries. It uses LangGraph for defining the agent's workflow, LangChain for LLM interaction, Gradio for the user interface, and the Model Context Protocol (MCP) for database interaction and schema retrieval.

## Features

*   **Natural Language Querying:** Users can ask questions in natural language (e.g., "show me all people", "how many cars are there?").
*   **LLM-Powered SQL Generation:** Uses a Large Language Model (LLM) via Ollama, OpenAI, or Gemini (configurable) to translate natural language into SQL queries.
*   **Dynamic Schema Retrieval:** Fetches the database schema dynamically at startup using an MCP call to the PostgreSQL server, ensuring the agent uses the current schema.
*   **Intent Classification:** Determines if the user's request is a `SELECT` query or an unsupported operation (INSERT, UPDATE, DELETE, etc.).
*   **SQL Validation:** Uses an LLM to perform basic validation and potential correction of the generated SQL.
*   **MCP Database Interaction:** Connects to a PostgreSQL database via a dedicated MCP server (`mcp_postgres`) to execute the generated `SELECT` queries using the `mcp` Python library. (Note: Currently uses a placeholder function in the UI).
*   **MCP Logging:** Prepares requests to log generated SQL and errors to a file (`/data/output.txt` within the filesystem container) via an MCP filesystem server. (Note: Also uses placeholder execution and overwrites the file).
*   **Gradio Web Interface:** Provides a user-friendly chat interface for interaction, displaying the conversation, agent steps (thoughts), generated SQL, prepared MCP requests, and query results.
*   **RAG (Retrieval-Augmented Generation):** Includes components for RAG using ChromaDB and OpenAI embeddings, initialized at startup but requires manual population.
*   **LangSmith Tracing (Optional):** Can be configured via environment variables to trace agent execution using LangSmith.
*   **Docker Support:** Includes `Dockerfile` and `docker-compose.yml` for containerized deployment (details need verification).

## Architecture & Workflow

The agent's logic is defined as a state machine using **LangGraph**. The core workflow involves the following steps (nodes):

1.  **Initialization (`app/main.py`):**
    *   Fetches the database schema dynamically via MCP (`fetch_dynamic_schema`).
    *   Initializes the RAG components (`initialize_rag`).

2.  **`nlp_agent_node` (`app/graph/nodes.py`):**
    *   Receives the user query.
    *   Uses an LLM to analyze the query against the *statically loaded* schema (`get_schema()` reading `schema.json` - **Note Discrepancy**).
    *   Attempts to extract relevant table names and refine the query via JSON output parsing.

3.  **`intent_classification_node` (`app/graph/nodes.py`):**
    *   Takes the (potentially refined) query.
    *   Uses keyword matching to classify the intent as `SELECT`, `UNSUPPORTED` (DML/DDL), or `OTHER`.

4.  **Conditional Routing (`decide_sql_generation_path` in `app/graph/builder.py`):**
    *   If `SELECT`, routes to `sql_generator`.
    *   If `UNSUPPORTED` or `OTHER`, routes directly to `prepare_mcp_requests_node`.

5.  **`sql_generation_node` (if intent is SELECT):**
    *   Receives the refined query, schema (static), and RAG context.
    *   Uses an LLM with a specific prompt to generate a single PostgreSQL `SELECT` statement.

6.  **`sql_validation_node` (if SQL was generated):**
    *   Receives the generated SQL.
    *   Uses an LLM with a strict prompt to check for common mistakes and potentially correct the query.

7.  **`prepare_mcp_requests_node`:**
    *   Takes the validated SQL (if generated) or handles cases where SQL generation was skipped/failed.
    *   Prepares request dictionaries for:
        *   Calling the `mcp_postgres` server's `query` tool (`prepare_mcp_sql_query_request`).
        *   Calling the `mcp_filesystem` server's `write_file` tool to log SQL/errors (`prepare_mcp_log_request`).
    *   Generates the final AI response message for the user.

8.  **Gradio UI & MCP Execution (`app/main.py`):**
    *   Displays the chat history, agent thoughts, SQL, and prepared MCP requests.
    *   Calls the `execute_mcp_tool` function (from `app/utils/mcp_utils.py`) to perform real MCP interactions for database queries. (Note: Logging via MCP is prepared but not executed by default).
    *   Displays query results (or errors) received from the MCP server in a DataFrame.

### Workflow Diagram (Mermaid)

```mermaid
graph TD
    subgraph Initialization
        direction LR
        I1[Start App] --> I2{fetch_dynamic_schema via MCP};
        I1 --> I3{initialize_rag};
    end

    subgraph Gradio Interaction
        direction TB
        A[User Query via Gradio] --> B(run_agent_graph_interface);
        B --> C{Invoke LangGraph};
        C --> D[nlp_agent_node (uses static schema)];
        D --> E[intent_classification_node];
        E --> F{decide_sql_generation_path};
        F -- Intent=SELECT --> G[sql_generation_node];
        F -- Intent=UNSUPPORTED/OTHER --> J[prepare_mcp_requests_node];
        G --> H[sql_validation_node];
        H --> J;
        J --> K{Return final_state};
        K --> L[Display MCP Requests];
        K --> M[execute_mcp_tool for Query];
        M --> N[Format Results/Errors];
        N --> O[Update Gradio UI];
    end

    I2 --> B; # Schema needed for agent run
    I3 --> B; # RAG needed for agent run
```

## Technologies Used

*   **Python:** Core programming language.
*   **LangChain:** Framework for LLM interaction, prompts, and chains.
*   **LangGraph:** Library for building stateful, multi-actor agent applications.
*   **Ollama / OpenAI / Gemini:** LLM providers (configurable via `app/llms/provider.py`).
*   **Gradio:** Framework for building the web UI.
*   **MCP (Model Context Protocol):** Protocol used for interacting with external services (DB, Filesystem).
    *   `mcp` Python Library: Used by the agent (`app/utils/mcp_utils.py`) to act as an MCP client.
    *   `mcp_postgres` Server (External): Expected to be running, providing a read-only `query` tool and schema access.
    *   `mcp_filesystem` Server (External): Expected to be running, providing a `write_file` tool.
*   **ChromaDB:** Vector database for RAG storage.
*   **OpenAI Embeddings:** Used for generating embeddings for RAG (requires API key).
*   **Pydantic:** For data validation and settings management (`app/config.py`).
*   **python-dotenv:** For loading environment variables from `.env`.
*   **Pandas:** For displaying query results in the Gradio UI.
*   **Docker / Docker Compose:** For containerization (requires running MCP servers separately or including them in compose).

## Setup & Running

1.  **Prerequisites:**
    *   Python 3.11+
    *   Docker and Docker Compose (Recommended for running MCP servers).
    *   Ollama installed and running *locally* if using Ollama models (e.g., `ollama pull llama3.2:latest`).
    *   Running MCP Servers:
        *   **PostgreSQL Server:** An MCP server connected to your target PostgreSQL database (e.g., the Node.js server from `modelcontextprotocol-servers/src/postgres`). Ensure the connection string used by the server is correct. The agent expects this server to be identified as `mcp_postgres`.
        *   **Filesystem Server:** An MCP server providing filesystem access (e.g., `mcp/filesystem` Docker image). The agent expects this server to be identified as `mcp_filesystem` and configured to allow writing to `/data/output.txt` (or adjust path in `app/utils/mcp_utils.py`).
    *   (Optional) OpenAI API Key set in `.env` if using OpenAI models or RAG embeddings.
    *   (Optional) Google API Key set in `.env` if using Gemini models.
    *   (Optional) LangSmith API Key and configuration set in `.env` for tracing.

2.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Yerdna1/MCP_SQL_AI_AGENT.git
    cd MCP_SQL_AI_AGENT
    ```

3.  **Configure Environment:**
    *   Copy `.env.example` to `.env` (if example exists) or create `.env`.
    *   Fill in necessary API keys (OpenAI, Google, LangSmith) if used.
    *   **Crucially:** Ensure the MCP server details (command, args, connection strings) in `app/utils/mcp_utils.py` (`MCP_POSTGRES_PARAMS`, etc.) match how your MCP servers are actually run.

4.  **Create Virtual Environment & Install Dependencies:**
    ```bash
    python -m venv venv
    # Activate (Windows PowerShell)
    .\venv\Scripts\Activate.ps1
    # Activate (macOS/Linux)
    # source venv/bin/activate
    pip install -r requirements.txt
    ```

5.  **Ensure `schema.json` is Present:**
    *   While the agent fetches schema dynamically, the `nlp_agent_node` currently still reads this static file. Ensure it reflects your target schema.

6.  **Run the Application:**
    *   **Ensure MCP Servers are Running:** Start your `mcp_postgres` and `mcp_filesystem` servers (e.g., via Docker Compose, separate terminals, etc.).
    *   **Run the Gradio App:**
        ```bash
        # Ensure any previous instances are stopped
        # On Windows: taskkill /F /IM python.exe /T
        # On macOS/Linux: pkill -f 'python -m app.main'

        # Run the Gradio app
        python -m app.main
        ```
    *   Access the UI via the URL provided (e.g., `http://localhost:7864`).

## Limitations & Future Work

*   **MCP Logging Not Executed:** While the agent prepares a request to log SQL/errors via an MCP filesystem server (`prepare_mcp_log_request`), the `app/main.py` UI logic does not currently execute this request. It only executes the database query request.
*   **Schema Discrepancy:** The application fetches the schema dynamically via MCP at startup (`fetch_dynamic_schema`), but the `nlp_agent_node` currently uses a static schema loaded from `schema.json` via `get_schema()`. This should be unified to consistently use the dynamically fetched schema within the graph state.
*   **DML/DDL Unsupported:** The agent only supports `SELECT` queries due to intent classification and the assumed read-only nature of the MCP postgres `query` tool.
*   **Log File Overwrite:** Logging via the MCP filesystem server uses `write_file`, which overwrites `output.txt` on each execution. True appending would require server support or a more complex read-modify-write pattern via MCP.
*   **RAG Population:** The RAG vector stores (ChromaDB) are initialized but require a mechanism to populate them with relevant SQL examples or documents (e.g., using the placeholder `kb_button` or automatically).
*   **NLP Robustness:** The `nlp_agent_node` relies on parsing JSON from LLM string output, which can be brittle. Using LLM functions/tools or more constrained output formats could improve reliability.
*   **Error Handling:** UI feedback for MCP connection errors, query execution errors, or graph failures could be more specific and user-friendly.
*   **Security:** Executing LLM-generated SQL carries inherent risks. Robust validation, potentially human-in-the-loop confirmation, and strict database permissions for the MCP server's connection are crucial.
*   **Configuration:** Some parameters like LLM model names, MCP server details, and RAG paths are hardcoded; moving them entirely to `settings` or `.env` would improve flexibility.
*   **Docker Compose:** The provided `docker-compose.yml` needs review to ensure it correctly sets up the application *and* the required MCP servers with appropriate volumes and environment variables.
