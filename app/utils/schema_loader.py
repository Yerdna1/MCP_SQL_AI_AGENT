import json
import logging
import os
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)

# --- Global cache for schema ---
_db_schema_info: Optional[Dict[str, Any]] = None
_schema_load_error: Optional[str] = None

SCHEMA_FILE_PATH = "schema.json" # Relative to WORKDIR /app, mounted via Docker

def load_schema_from_file(force_reload: bool = False) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Loads database schema information from a JSON file.
    Uses a simple in-memory cache unless force_reload is True.

    Args:
        force_reload: If True, bypasses the cache and reloads from the file.

    Returns:
        A tuple containing:
            - The loaded schema dictionary (or None if failed).
            - An error message string (or None if successful).
    """
    global _db_schema_info, _schema_load_error

    if not force_reload and (_db_schema_info is not None or _schema_load_error is not None):
        logger.debug("Returning cached schema info/error.")
        return _db_schema_info, _schema_load_error

    logger.info(f"Attempting to load schema from {SCHEMA_FILE_PATH}...")
    try:
        if os.path.exists(SCHEMA_FILE_PATH):
            with open(SCHEMA_FILE_PATH, 'r', encoding='utf-8') as f:
                _db_schema_info = json.load(f)
            logger.info(f"Schema loaded successfully from {SCHEMA_FILE_PATH}: Keys = {list(_db_schema_info.keys()) if _db_schema_info else 'None'}")
            _schema_load_error = None # Reset error on successful load
        else:
            _schema_load_error = f"Schema file '{SCHEMA_FILE_PATH}' not found. Please ensure it's mounted correctly."
            logger.error(_schema_load_error)
            _db_schema_info = None
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse schema file {SCHEMA_FILE_PATH}: Invalid JSON - {e}", exc_info=True)
        _schema_load_error = f"Error loading schema: Invalid JSON in {SCHEMA_FILE_PATH}."
        _db_schema_info = None
    except Exception as e:
        logger.error(f"Failed to load schema file {SCHEMA_FILE_PATH}: {e}", exc_info=True)
        _schema_load_error = f"Error loading schema: {type(e).__name__}: {e}"
        _db_schema_info = None

    logger.info("Schema loading process finished.")
    return _db_schema_info, _schema_load_error

def get_schema() -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Gets the cached schema info or loads it if not already loaded."""
    if _db_schema_info is None and _schema_load_error is None:
        # Attempt load on first call if not initialized
        return load_schema_from_file()
    return _db_schema_info, _schema_load_error
