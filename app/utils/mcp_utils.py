import datetime
import logging
from typing import Optional, Dict, Any, Tuple # Add Tuple

logger = logging.getLogger(__name__)

def prepare_mcp_sql_query_request(sql_query: str) -> Dict[str, Any]:
    """
    Prepares the arguments dictionary for an MCP postgres.query request.
    This dictionary should be displayed to the user, who then uses their
    MCP client (e.g., VS Code extension) to execute the tool call.

    Args:
        sql_query: The SQL query string to be executed.

    Returns:
        A dictionary representing the MCP tool call request.
    """
    logger.info(f"Preparing MCP query request for SQL: {sql_query[:100]}...") # Log snippet
    return {
        "server_name": "mcp_postgres", # Matches service name in docker-compose.yml
        "tool_name": "query",
        "arguments": {"sql": sql_query}
    }

def prepare_mcp_log_request(sql: str, error: Optional[str] = None) -> Dict[str, Any]:
    """
    Prepares the arguments dictionary for an MCP filesystem.append_file request
    to log the SQL query.
    This dictionary should be displayed to the user for execution via their MCP client.

    Args:
        sql: The SQL query string that was generated/executed.
        error: An optional error message associated with the query.

    Returns:
        A dictionary representing the MCP tool call request.
    """
    log_content = f"-- Timestamp: {datetime.datetime.now().isoformat()} --\n"
    log_content += f"SQL: {sql}\n"
    if error:
        log_content += f"Error: {error}\n"
    log_content += "-" * 20 + "\n\n" # Add separator after each entry

    # Use path from settings
    log_file_path = settings.mcp_log_file_path
    logger.info(f"Preparing MCP log append request for '{log_file_path}'")
    return {
        "server_name": "mcp_filesystem", # Matches service name in docker-compose.yml
        "tool_name": "append_file", # Use the append tool
        "arguments": {
            "path": log_file_path, # Path inside the filesystem container volume mount
            "content": log_content # Content to append
        }
    }

def prepare_mcp_read_file_request(file_path: str) -> Dict[str, Any]:
    """
    Prepares the arguments dictionary for an MCP filesystem.read_file request.

    Args:
        file_path: The path of the file to read inside the MCP filesystem server's context.

    Returns:
        A dictionary representing the MCP tool call request.
    """
    # Note: file_path here is relative to the server root defined in settings
    logger.info(f"Preparing MCP request to read file: {file_path}")
    return {
        "server_name": "mcp_filesystem",
        "tool_name": "read_file",
        "arguments": {"path": file_path} # Assume file_path is already correct relative path
    }

def prepare_mcp_save_sql_request(sql_query: str) -> Dict[str, Any]:
    """
    Prepares the arguments dictionary for an MCP filesystem.append_file request
    to save a successfully executed SQL query.

    Args:
        sql_query: The SQL query string to save.

    Returns:
        A dictionary representing the MCP tool call request.
    """
    # Add a newline before appending to keep queries separate
    content_to_append = f"{sql_query.strip()};\n"

    # Use path from settings
    save_sql_path = settings.mcp_saved_sql_path
    logger.info(f"Preparing MCP request to save successful SQL to '{save_sql_path}'")
    return {
        "server_name": "mcp_filesystem",
        "tool_name": "append_file",
        "arguments": {
            "path": save_sql_path, # Dedicated file for successful queries
            "content": content_to_append
        }
    }


# Example of how to prepare a GDrive search request (for KB population)
def prepare_mcp_gdrive_search_request(search_query: str) -> Dict[str, Any]:
    """Prepares the arguments for an MCP gdrive.search request."""
    logger.info(f"Preparing MCP GDrive search request for query: {search_query}")
    return {
        "server_name": "mcp_gdrive", # Matches service name in docker-compose.yml
        "tool_name": "search",
        "arguments": {"query": search_query}
    }

# Example of how to prepare a GDrive file read request (for KB population)
def prepare_mcp_gdrive_read_request(file_id: str) -> Dict[str, Any]:
    """Prepares the arguments for an MCP gdrive resource read request."""
    logger.info(f"Preparing MCP GDrive read request for file ID: {file_id}")
    # Note: Resource access uses access_mcp_resource, not use_mcp_tool
    # This function returns the parameters needed for that structure.
    return {
        "server_name": "mcp_gdrive",
        "uri": f"gdrive:///{file_id}"
    }

# --- Real MCP Tool Execution ---
import sys
import asyncio
import json # Add json import
import os # Need os for path join
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
# Assuming mcp.types is available if needed, but call_tool might not need it directly
from ..config import settings # Import settings

# --- Dynamically build MCP Server Parameters from settings ---

# Postgres Params
MCP_POSTGRES_PARAMS = StdioServerParameters(
    command=settings.mcp_postgres_command,
    args=[
        settings.mcp_postgres_script_path,
        settings.mcp_postgres_conn_string.get_secret_value() # Get value from SecretStr
    ],
    env=None # Or specific env vars if needed
)

# Filesystem Params
# Construct the mount argument dynamically
mount_source_dir = f"{settings.mcp_filesystem_mount_source_prefix}_{datetime.date.today().strftime('%Y%m%d')}"
mount_arg = f"type=bind,src={mount_source_dir},dst={settings.mcp_filesystem_mount_target}"

MCP_FILESYSTEM_PARAMS = StdioServerParameters(
    command=settings.mcp_filesystem_command,
    args=[
        "run", "-i", "--rm",
        "--mount", mount_arg,
        settings.mcp_filesystem_image,
        settings.mcp_filesystem_server_root
    ],
    env=None
)

# GDrive Params (NPX based)
MCP_GDRIVE_PARAMS = StdioServerParameters(
    command="npx",
    args=["-y", settings.mcp_gdrive_npx_package],
    env=None # Assumes OAuth keys/credentials are in default locations relative to execution
)
# --- End MCP Server Params ---


async def execute_mcp_tool(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes an MCP tool call using the mcp library via STDIO.
    """
    server_name = request.get("server_name")
    tool_name = request.get("tool_name")
    args = request.get("arguments", {})

    logger.info(f"Executing MCP Call: Server='{server_name}', Tool='{tool_name}', Args='{args}'")

    # Select the correct server parameters
    if server_name == "mcp_postgres": # Match the name used in prepare_mcp_sql_query_request
         server_params = MCP_POSTGRES_PARAMS
    elif server_name == "mcp_filesystem": # Match the name used in prepare_mcp_log_request
         server_params = MCP_FILESYSTEM_PARAMS
    elif server_name == settings.mcp_gdrive_server_name: # Use name from settings
         server_params = MCP_GDRIVE_PARAMS
    else:
         error_msg = f"Unknown MCP server name specified in request: {server_name}"
         logger.error(error_msg)
         return {"success": False, "error": error_msg}

    try:
        # Use AsyncExitStack to manage the connection lifecycle
        async with AsyncExitStack() as stack:
            logger.debug(f"Connecting to MCP server '{server_name}' via STDIO...")
            stdio_transport = await stack.enter_async_context(stdio_client(server_params))
            stdio, write = stdio_transport
            logger.debug("STDIO transport established.")

            session = await stack.enter_async_context(ClientSession(stdio, write))
            logger.debug("Client session created.")

            await session.initialize()
            logger.debug("MCP Connection initialized.")

            # --- Actual MCP client call ---
            # Pass tool_name and args as positional arguments
            mcp_response = await session.call_tool(tool_name, args)
            # --- End of actual call ---

            logger.info(f"MCP Response received: Type={type(mcp_response)}")
            # Log content snippet for debugging
            if hasattr(mcp_response, 'content') and mcp_response.content:
                 logger.debug(f"MCP Response content snippet: {str(mcp_response.content[0])[:200]}")


            # --- Parse the actual MCP response ---
            # Assuming response has a 'content' attribute which is a list of content objects
            # And content objects have a 'text' attribute for simple text results
            # Adjust parsing based on actual mcp library response structure for tool calls
            if mcp_response and hasattr(mcp_response, 'content') and mcp_response.content:
                 # Attempt to parse JSON if content looks like it
                 try:
                      # Assuming the first content part holds the result text
                      result_text = getattr(mcp_response.content[0], 'text', None)
                      if result_text:
                           parsed_result = json.loads(result_text)
                           logger.info("Successfully parsed MCP result as JSON.")
                           # Check if the parsed result is the list of dicts format
                           if isinstance(parsed_result, list):
                                return {"success": True, "result": parsed_result}
                           else:
                                # Return the parsed JSON even if not list (e.g., simple status)
                                return {"success": True, "result": parsed_result}
                      else:
                           logger.warning("MCP response content had no 'text' attribute.")
                           # Fallback: return raw content object if no text? Or handle differently.
                           return {"success": True, "result": mcp_response.content}

                 except json.JSONDecodeError:
                      logger.warning("MCP result content was not valid JSON. Returning as raw text.")
                      # If not JSON, return the raw text content
                      return {"success": True, "result": result_text or str(mcp_response.content)}
                 except Exception as parse_e:
                      logger.error(f"Error processing MCP response content: {parse_e}", exc_info=True)
                      return {"success": False, "error": f"Error processing MCP response: {parse_e}"}
            else:
                 # Handle cases where response might be successful but have no content (e.g., write operations)
                 logger.info("MCP call successful but returned no content.")
                 return {"success": True, "result": "Operation successful (no content returned)."}

            # Note: Need to refine error handling based on how the mcp library signals errors
            # For now, assuming lack of content might not always be an error.

    except ConnectionRefusedError as conn_e:
         error_msg = f"MCP Connection Error for '{server_name}': {conn_e}. Is the server running?"
         logger.error(error_msg, exc_info=True)
         return {"success": False, "error": error_msg}
    except Exception as e:
        error_msg = f"Error during MCP execution for '{server_name}': {type(e).__name__}: {e}"
        logger.error(error_msg, exc_info=True)
        return {"success": False, "error": error_msg}


# --- Dynamic Schema Fetching ---

# SQL query to get table and column information
SCHEMA_QUERY = """
SELECT
    t.table_name,
    c.column_name,
    c.data_type,
    c.is_nullable,
    kcu.constraint_name AS pk_constraint_name,
    tc.constraint_type AS pk_constraint_type
FROM
    information_schema.tables t
JOIN
    information_schema.columns c ON t.table_name = c.table_name AND t.table_schema = c.table_schema
LEFT JOIN
    information_schema.key_column_usage kcu ON c.table_name = kcu.table_name
                                            AND c.column_name = kcu.column_name
                                            AND c.table_schema = kcu.table_schema
LEFT JOIN
    information_schema.table_constraints tc ON kcu.constraint_name = tc.constraint_name
                                             AND kcu.table_schema = tc.table_schema
                                             AND tc.constraint_type = 'PRIMARY KEY'
WHERE
    t.table_schema = 'public'  -- Or specify other schemas if needed
    AND t.table_type = 'BASE TABLE'
ORDER BY
    t.table_name,
    c.ordinal_position;
"""

async def fetch_dynamic_schema() -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Fetches the current database schema dynamically using MCP.

    Returns:
        A tuple containing:
            - The schema dictionary (or None if failed).
            - An error message string (or None if successful).
    """
    logger.info("Attempting to fetch schema dynamically via MCP...")
    schema_request = prepare_mcp_sql_query_request(SCHEMA_QUERY)
    mcp_response = await execute_mcp_tool(schema_request)

    if not mcp_response.get("success"):
        error_msg = f"MCP call failed during schema fetch: {mcp_response.get('error', 'Unknown MCP error')}"
        logger.error(error_msg)
        return None, error_msg

    raw_result = mcp_response.get("result")
    if not isinstance(raw_result, list):
        error_msg = f"Unexpected result format from MCP schema query: {type(raw_result)}. Expected list."
        logger.error(error_msg)
        return None, error_msg

    # Process the list of dictionaries into the desired schema format
    # Expected format: { "table_name": {"columns": ["col1", "col2"], "primary_key": "col1"}, ... }
    db_schema_info: Dict[str, Any] = {}
    try:
        for row in raw_result:
            table_name = row.get("table_name")
            column_name = row.get("column_name")
            data_type = row.get("data_type")
            # is_nullable = row.get("is_nullable") # Could add this info
            pk_type = row.get("pk_constraint_type")

            if not table_name or not column_name:
                continue # Skip rows with missing essential info

            if table_name not in db_schema_info:
                db_schema_info[table_name] = {"columns": [], "primary_key": None}

            db_schema_info[table_name]["columns"].append(f"{column_name} ({data_type})") # Add type info

            if pk_type == 'PRIMARY KEY':
                 # Handle composite keys? For now, assume single PK or take the first one found.
                 if db_schema_info[table_name]["primary_key"] is None:
                      db_schema_info[table_name]["primary_key"] = column_name
                 else:
                      # Log if multiple PK columns are found for a table (composite key)
                      logger.warning(f"Table '{table_name}' appears to have a composite primary key. Storing only first PK column found ('{db_schema_info[table_name]['primary_key']}').")


        logger.info(f"Dynamically fetched schema: Keys = {list(db_schema_info.keys())}")
        return db_schema_info, None

    except Exception as e:
        error_msg = f"Error processing fetched schema data: {e}"
        logger.error(error_msg, exc_info=True)
        return None, error_msg


# --- MCP Resource Access ---
async def access_mcp_resource(server_name: str, resource_uri: str) -> Dict[str, Any]:
    """
    Accesses a resource provided by an MCP server.
    Currently placeholder - needs actual implementation using mcp library's
    resource access methods if available, or potentially adapting execute_mcp_tool
    if resource access is exposed via a 'read_resource' tool convention.

    Args:
        server_name: The name of the MCP server providing the resource.
        resource_uri: The URI of the resource to access.

    Returns:
        A dictionary containing success status and result/error.
    """
    logger.warning(f"Placeholder: Attempting to access MCP resource. Server='{server_name}', URI='{resource_uri}'")
    # This requires the mcp library's specific API for resource access,
    # which might differ from tool calls.
    # For now, return a placeholder error indicating it's not implemented.
    return {
        "success": False,
        "error": f"Resource access for server '{server_name}' not implemented in this utility function."
        # In a real implementation, you'd use session.read_resource(uri=resource_uri)
        # after establishing a connection similar to execute_mcp_tool.
    }
