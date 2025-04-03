import logging
import os
from typing import Optional, Dict, Any, List, Tuple, Sequence # Add Sequence

import chromadb
from langchain_openai import OpenAIEmbeddings # Assuming OpenAI for embeddings
from langchain_core.embeddings import Embeddings as LangchainEmbeddingsCore # Rename to avoid clash
from pydantic import BaseModel # Import BaseModel for type hinting settings
# Import ChromaDB types for the wrapper
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings as ChromaEmbeddings

# Remove direct import of settings, it will be passed in
# from ..config import settings

logger = logging.getLogger(__name__)

# --- RAG Component Initialization ---
# These will be initialized by initialize_rag()
# Use the renamed import for the Langchain object
embeddings: Optional[LangchainEmbeddingsCore] = None
sql_collection: Optional[chromadb.Collection] = None
gdrive_collection: Optional[chromadb.Collection] = None
rag_initialized: bool = False
initialization_error: Optional[str] = None

# Constants - Collection names remain constant
SQL_COLLECTION_NAME = "sql_examples"
GDRIVE_COLLECTION_NAME = "gdrive_docs"

# --- Add Wrapper Class ---
class LangchainEmbeddingFunctionWrapper(EmbeddingFunction):
    """Simple wrapper to adapt Langchain Embeddings to ChromaDB's interface."""
    def __init__(self, langchain_embeddings: LangchainEmbeddingsCore):
        self._langchain_embeddings = langchain_embeddings

    # Implement the __call__ method expected by ChromaDB
    def __call__(self, input: Documents) -> ChromaEmbeddings:
        # Pass the documents to the Langchain embedding model
        return self._langchain_embeddings.embed_documents(input)
# --- End Wrapper Class ---


# Accept settings object as argument
def initialize_rag(app_settings: BaseModel) -> Tuple[bool, Optional[str]]:
    """
    Initializes ChromaDB client, embedding function, and collections.
    Sets global variables for other functions in this module to use.

    Returns:
        A tuple: (success_status, error_message)
    """
    global embeddings, sql_collection, gdrive_collection, rag_initialized, initialization_error

    if rag_initialized:
        logger.info("RAG already initialized.")
        return True, None

    # Get config values from the passed-in settings object
    chroma_db_path = app_settings.chroma_db_path
    embedding_model_name = app_settings.embedding_model # Get embedding model name
    logger.info(f"Initializing RAG components. Chroma Path: {chroma_db_path}, Embedding Model: {embedding_model_name}")
    try:
        # 1. Ensure ChromaDB path exists
        if not os.path.exists(chroma_db_path):
            os.makedirs(chroma_db_path, exist_ok=True)
            logger.info(f"Created ChromaDB directory: {chroma_db_path}")

        # 2. Initialize ChromaDB client
        chroma_client = chromadb.PersistentClient(path=chroma_db_path) # Use local variable

        # 3. Initialize Embeddings (Requires OpenAI API Key)
        # Use the passed-in settings object
        if app_settings.openai_api_key:
            # Keep the initialized Langchain embeddings object
            langchain_embeddings = OpenAIEmbeddings(
                model=embedding_model_name, # Use model name from settings
                api_key=app_settings.openai_api_key.get_secret_value()
            )
            # Store it globally if needed elsewhere, maybe rename global var?
            embeddings = langchain_embeddings
            # Create the wrapper for ChromaDB
            chroma_embedding_function = LangchainEmbeddingFunctionWrapper(langchain_embeddings)
            logger.info(f"Initialized OpenAI Embeddings: {embedding_model_name} and ChromaDB wrapper.")
        else:
            error_msg = "OpenAI API Key not found in settings. Cannot initialize embeddings."
            logger.error(error_msg)
            initialization_error = error_msg
            rag_initialized = False
            return False, error_msg

        # 4. Get or create ChromaDB collections
        # Use the embedding function directly if ChromaDB version supports it,
        # otherwise, embeddings need to be generated before adding.
        # Assuming newer ChromaDB versions that handle embedding function internally.
        # Use the WRAPPER class instance for ChromaDB
        sql_collection = chroma_client.get_or_create_collection(
            name=SQL_COLLECTION_NAME,
            embedding_function=chroma_embedding_function # Use the wrapper
            # metadata={"hnsw:space": "cosine"} # Optional: configure index params
        )
        logger.info(f"ChromaDB SQL examples collection '{SQL_COLLECTION_NAME}' loaded/created. Count: {sql_collection.count()}")

        gdrive_collection = chroma_client.get_or_create_collection(
            name=GDRIVE_COLLECTION_NAME,
            embedding_function=chroma_embedding_function # Use the wrapper
            # metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"ChromaDB GDrive docs collection '{GDRIVE_COLLECTION_NAME}' loaded/created. Count: {gdrive_collection.count()}")

        rag_initialized = True
        initialization_error = None
        logger.info("RAG components initialized successfully.")
        return True, None

    except Exception as e:
        error_msg = f"Failed to initialize RAG components: {e}"
        logger.error(error_msg, exc_info=True)
        initialization_error = error_msg
        rag_initialized = False
        # Reset globals on failure
        embeddings = None
        sql_collection = None
        gdrive_collection = None
        return False, error_msg


def retrieve_rag_context(query: str, n_results: int = 3) -> dict:
    """
    Retrieves context from ChromaDB collections based on the user query.

    Args:
        query: The user's query string.
        n_results: The number of results to retrieve from each collection.

    Returns:
        A dictionary containing lists of retrieved documents:
        {"sql_examples": [...], "gdrive_docs": [...]}
    """
    if not rag_initialized or not embeddings or sql_collection is None or gdrive_collection is None:
        logger.warning(f"RAG not initialized or components missing. Skipping retrieval. Status: {rag_initialized}, Error: {initialization_error}")
        return {"sql_examples": [], "gdrive_docs": []}

    logger.info(f"Retrieving RAG context for query: {query[:100]}...")
    retrieved_sql = []
    retrieved_gdrive = []

    try:
        # Embed the user query (done by ChromaDB internally if embedding_function is set)
        # query_embedding = embeddings.embed_query(query) # Not needed if handled by ChromaDB

        # Retrieve from SQL examples collection
        sql_results = sql_collection.query(
            query_texts=[query], # Pass query text directly
            n_results=n_results,
            include=["documents"] # Only need the document content
        )
        retrieved_sql = [doc for doc_list in sql_results.get('documents', [[]]) for doc in doc_list]
        logger.info(f"Retrieved {len(retrieved_sql)} SQL examples.")

        # Retrieve from Google Drive documents collection
        gdrive_results = gdrive_collection.query(
            query_texts=[query], # Pass query text directly
            n_results=n_results,
            include=["documents"]
        )
        retrieved_gdrive = [doc for doc_list in gdrive_results.get('documents', [[]]) for doc in doc_list]
        logger.info(f"Retrieved {len(retrieved_gdrive)} GDrive chunks.")

    except Exception as e:
        logger.error(f"Error during RAG retrieval: {e}", exc_info=True)
        # Return empty lists on error, but log the issue
        return {"sql_examples": [], "gdrive_docs": []}

    return {"sql_examples": retrieved_sql, "gdrive_docs": retrieved_gdrive}

# Example function to add data (e.g., successful SQL queries)
# This would be called from the graph or another process
def add_sql_example_to_kb(nl_query: str, sql_query: str):
    """Adds a natural language query and its successful SQL to the KB."""
    if not rag_initialized or sql_collection is None:
        logger.error("Cannot add SQL example: RAG not initialized.")
        return

    try:
        # Combine NL and SQL for context, or store separately in metadata
        doc_content = f"User Query: {nl_query}\nGenerated SQL: {sql_query}"
        doc_id = str(uuid.uuid4()) # Generate unique ID

        sql_collection.add(
            documents=[doc_content],
            ids=[doc_id]
            # metadatas=[{"nl_query": nl_query, "sql_query": sql_query}] # Optional metadata
        )
        logger.info(f"Added SQL example to KB with ID: {doc_id}")
    except Exception as e:
        logger.error(f"Failed to add SQL example to KB: {e}", exc_info=True)
