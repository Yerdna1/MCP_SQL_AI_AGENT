# PostgreSQL AI Agent (LangGraph + MCP Mode)

This project implements an AI agent capable of interacting with a PostgreSQL database based on natural language queries. It uses LangGraph for defining the agent's workflow, LangChain for LLM interaction, Gradio for the user interface, and the Model Context Protocol (MCP) for database interaction and schema retrieval.

## Features

*   **Natural Language Querying:** Users can ask questions in natural language (e.g., "show me all people", "how many cars are there?").
*   **LLM-Powered SQL Generation:** Uses a Large Language Model (LLM) via Ollama, OpenAI, or Gemini (configurable) to translate natural language into SQL queries.
*   **Dynamic Schema Retrieval:** Fetches the database schema dynamically at startup using an MCP call to the PostgreSQL server, ensuring the agent uses the current schema.
*   **Intent Classification:** Determines if the user's request is a `SELECT` query or an unsupported operation (INSERT, UPDATE, DELETE, etc.).
*   **SQL Validation:** Uses an LLM to perform basic validation and potential correction of the generated SQL.
*   **MCP Database Interaction:** Connects to a PostgreSQL database via a dedicated MCP server (`mcp_postgres`) to execute the generated `SELECT` queries using the `mcp` Python library.
*   **MCP Logging:** Appends generated SQL and errors to a log file (`/data/output.txt` within the filesystem container) via the `append_file` tool of an MCP filesystem server.
*   **Automatic SQL Saving:** Successfully executed SQL queries are automatically appended to `/data/successful_queries.sql` via MCP, creating a source for RAG knowledge base population.
*   **Gradio Web Interface:** Provides a user-friendly chat interface for interaction, displaying the conversation, agent steps (thoughts), generated SQL, prepared MCP requests, and query results. Includes a button to update the RAG SQL knowledge base.
*   **RAG (Retrieval-Augmented Generation):** Includes components for RAG using ChromaDB and OpenAI embeddings, initialized at startup. The SQL examples knowledge base can be populated from the saved `successful_queries.sql` file using the "Update SQL KB" button in the UI (requires MCP filesystem server access).
*   **LangSmith Tracing (Optional):** Can be configured via environment variables to trace agent execution using LangSmith.
*   **Docker Support:** Includes `Dockerfile` and `docker-compose.yml` for containerized deployment (details need verification).

## Architecture & Workflow

The agent's logic is defined as a state machine using **LangGraph**. The core workflow involves the following steps (nodes):

1.  **Initialization (`app/main.py`):**
    *   Fetches the database schema dynamically via MCP (`fetch_dynamic_schema`).
    *   Initializes the RAG components (`initialize_rag`).

2.  **`nlp_agent_node` (`app/graph/nodes.py`):**
    *   Receives the user query.
    *   Uses an LLM to analyze the query against the dynamically fetched schema (from agent state).
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
    *   Calls the `execute_mcp_tool` function (from `app/utils/mcp_utils.py`) to perform real MCP interactions for:
        *   Database queries (`mcp_postgres` server) **after user approval via the 'Execute Approved SQL' button**.
        *   Appending SQL/errors to log file (`mcp_filesystem` server's `append_file` tool) **after successful query execution**.
        *   Appending successfully executed SQL to `successful_queries.sql` (`mcp_filesystem` server's `append_file` tool) **after successful query execution**.
    *   Displays query results (or errors) received from the MCP server in a DataFrame.
    *   Adds logging and SQL saving status/errors to the execution status display.

### Workflow Diagram (Mermaid)

```mermaid
graph TD
    subgraph Initialization
        direction LR
        I1[Start App] --> I2{fetch_dynamic_schema via MCP}
        I1 --> I3{initialize_rag}
    end

    subgraph Gradio Interaction
        direction TB
        A[User Query via Gradio] --> B(run_agent_graph_interface)
        B --> C{Invoke LangGraph}
        C --> D[nlp_agent_node]
        D --> E[intent_classification_node]
        E --> F{decide_sql_generation_path}
        F -- Intent=SELECT --> G[sql_generation_node]
        F -- Intent=UNSUPPORTED/OTHER --> J[prepare_mcp_requests_node]
        G --> H[sql_validation_node]
        H --> J
        J --> K{Return final_state}
        K --> L[Display SQL & Requests]
        L --> Q{User Clicks 'Execute Approved SQL'}
        Q --> M(execute_approved_sql_handler) %% Trigger handler
        
        subgraph execute_approved_sql_handler
            direction TB
            M --> M1[execute_mcp_tool for Query]
            M1 -- Success --> N[execute_mcp_tool for Log]
            N -- Success --> R[execute_mcp_tool for Save SQL]
            R --> P[Format Results/Errors]
            M1 -- Failure --> P
            N -- Failure --> R
        end

        P --> O[Update Gradio UI]
    end

    I2 --> B
    I3 --> B
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
*   **python-dotenv:** For loading environment variables from `.env` (see `.env.example`).
*   **Pandas:** For displaying query results in the Gradio UI.
*   **Docker / Docker Compose:** For containerizing the main application. MCP servers are launched directly by the Python application using configured commands (e.g., `node`, `docker run`, `npx`).

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
    *   Fill in necessary API keys (OpenAI, Google, LangSmith).
    *   Review and adjust default model names (`OLLAMA_MODEL`, `OPENAI_MODEL`, `GEMINI_MODEL`, `EMBEDDING_MODEL`), the list of available LLMs (`AVAILABLE_LLMS`), ChromaDB path (`CHROMA_DB_PATH`), MCP server details, and MCP filesystem paths if needed.

4.  **Create Virtual Environment & Install Dependencies:**
    ```bash
    python -m venv venv
    # Activate (Windows PowerShell)
    .\venv\Scripts\Activate.ps1
    # Activate (macOS/Linux)
    # source venv/bin/activate
    pip install -r requirements.txt
    ```

5.  **Run the Application:**
    *   **Ensure MCP Server Prerequisites are Met:**
        *   Node.js installed (for `mcp_postgres`).
        *   Docker installed and running (for `mcp_filesystem`).
        *   NPX available (usually comes with Node.js) (for `mcp_gdrive`).
        *   GDrive server authenticated separately if using GDrive features (see GDrive server docs).
    *   **Run the Gradio App (Docker Compose Recommended):**
        *   Ensure your `.env` file is correctly configured.
        *   Run: `docker-compose up --build`
        *   The application service will start, and the Python code within it will attempt to launch the configured MCP servers via `stdio_client`. Check the container logs for any errors related to MCP server startup.
    *   **Alternatively, Run Locally (Requires Manual MCP Server Startup):**
        *   Start the `mcp_postgres`, `mcp_filesystem`, and `mcp_gdrive` servers manually in separate terminals according to their respective documentation and the paths/commands configured in your `.env` file.
        *   Run the Gradio app directly:
            ```bash
        # Ensure any previous instances are stopped
        # On Windows: taskkill /F /IM python.exe /T
        # On macOS/Linux: pkill -f 'python -m app.main'

        # Run the Gradio app
        python -m app.main
        ```
    *   Access the UI via the URL provided (e.g., `http://localhost:7864`).

## Limitations & Future Work

*   **DML/DDL Unsupported:** The agent only supports `SELECT` queries due to intent classification and the assumed read-only nature of the MCP postgres `query` tool.
*   **RAG GDrive Population:** Populating the RAG knowledge base from Google Drive documents requires manual steps using an MCP client, as detailed in the UI when clicking the "Prepare GDrive KB Population" button. The `populate_gdrive_kb` function only prepares the initial search request.
*   **Error Handling:** UI feedback for MCP connection errors, query execution errors, or graph failures could be more specific and user-friendly.
*   **Security:** The human-in-the-loop step adds a layer of safety, but ensure the MCP server's database connection uses appropriately restricted permissions.
*   **MCP Server Management:** The application now directly manages MCP server processes via `stdio_client`. Ensure the commands and paths in the settings (`.env`) are correct for your environment. Errors during MCP server startup will appear in the application's logs.
