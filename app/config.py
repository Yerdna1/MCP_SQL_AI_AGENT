# app/config.py
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field, PostgresDsn, SecretStr
from typing import Optional

# Define path to .env file relative to this config file
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
# Explicitly load .env *before* Pydantic model instantiation
load_dotenv(dotenv_path=dotenv_path, override=True) # Use override=True just in case

# --- Manually read potentially problematic keys AFTER load_dotenv ---
manual_openai_key = os.getenv('OPENAI_API_KEY')
manual_google_key = os.getenv('GOOGLE_API_KEY')
manual_langsmith_key = os.getenv('LANGSMITH_API_KEY')
# ---

# --- Convert manually loaded keys to SecretStr if they exist ---
secret_openai_key = SecretStr(manual_openai_key) if manual_openai_key else None
secret_google_key = SecretStr(manual_google_key) if manual_google_key else None
secret_langsmith_key = SecretStr(manual_langsmith_key) if manual_langsmith_key else None
# ---

class Settings(BaseModel):
    """Application settings"""

    # LLM API Keys (Optional) - Initialize with SecretStr objects
    openai_api_key: Optional[SecretStr] = secret_openai_key
    google_api_key: Optional[SecretStr] = secret_google_key

    # LangSmith Configuration (Optional) - Use Field for non-SecretStr or less problematic ones
    langchain_tracing_v2: bool = Field(False, env='LANGCHAIN_TRACING_V2')
    langchain_endpoint: Optional[str] = Field(None, env='LANGCHAIN_ENDPOINT')
    langsmith_api_key: Optional[SecretStr] = secret_langsmith_key # Initialize with SecretStr
    langchain_project: Optional[str] = Field(None, env='LANGCHAIN_PROJECT')

    # Gmail Configuration (Optional - paths relative to container WORKDIR /app)
    gmail_credentials_file: str = Field("credentials.json", env='GMAIL_CREDENTIALS_FILE')
    gmail_token_file: str = Field("token.json", env='GMAIL_TOKEN_FILE')

    # Output file for SQL logs
    sql_output_file: str = "output.txt" # Relative to WORKDIR /app

    # ChromaDB path (inside container)
    chroma_db_path: str = Field("/app/chroma_db", env='CHROMA_DB_PATH')

    # Model Names (allow override via environment)
    ollama_model: str = Field("llama3.2:latest", env='OLLAMA_MODEL')
    embedding_model: str = Field("text-embedding-ada-002", env='EMBEDDING_MODEL')

    class Config:
        # Keep env_file for potential fallback or other variables, but load_dotenv should take precedence
        # env_file = dotenv_path # Comment out to rely solely on os.getenv for keys above
        env_file_encoding = 'utf-8'
        # Pydantic v2 settings
        # extra = 'ignore' # If using Pydantic v2, uncomment if needed

# Create a single instance of settings to be imported elsewhere
settings = Settings()


# Optional: Setup LangSmith tracing if configured
if settings.langchain_tracing_v2 and settings.langsmith_api_key and settings.langchain_endpoint:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key.get_secret_value()
    if settings.langchain_project:
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
    print("LangSmith tracing enabled.")
else:
    print("LangSmith tracing not configured or disabled.")
