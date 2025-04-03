import logging
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama # Import ChatOllama instead of OllamaLLM
from langchain_core.language_models.chat_models import BaseChatModel

# Assuming settings are imported where this function is called, or passed explicitly
# For simplicity here, let's assume direct import (though dependency injection is better)
from ..config import settings

logger = logging.getLogger(__name__)

def get_llm_instance(llm_name: str) -> BaseChatModel:
    """
    Returns an instance of the specified LLM based on configuration.

    Args:
        llm_name: The name of the LLM selected (e.g., "Ollama (Local)", "OpenAI (API)").

    Returns:
        An instance of the Langchain chat model.

    Raises:
        ValueError: If the selected LLM is not available or configured.
    """
    logger.info(f"Attempting to get LLM instance for: {llm_name}")
    if llm_name == "Ollama (Local)":
        try:
            # Use the model name from settings
            ollama_model = settings.ollama_model
            logger.info(f"Initializing ChatOllama with model: {ollama_model}")
            # Instantiate ChatOllama
            # Add format="json" if needed for specific models, or rely on with_structured_output
            return ChatOllama(model=ollama_model, temperature=0) # Keep temperature if desired
        except Exception as e:
            logger.error(f"Failed to initialize ChatOllama: {e}", exc_info=True)
            raise ValueError(f"Failed to initialize Ollama Chat model: {e}")

    elif llm_name == "OpenAI (API)":
        if settings.openai_api_key:
            # Use model name from settings
            openai_model = settings.openai_model
            logger.info(f"Initializing OpenAI with model: {openai_model}")
            return ChatOpenAI(model=openai_model, api_key=settings.openai_api_key.get_secret_value())
        else:
            logger.error("OpenAI API Key not found in settings.")
            raise ValueError("OpenAI API Key not configured.")

    elif llm_name == "Gemini (API)":
        if settings.google_api_key:
            # Use model name from settings
            gemini_model = settings.gemini_model
            logger.info(f"Initializing Gemini with model: {gemini_model}")
            return ChatGoogleGenerativeAI(model=gemini_model, google_api_key=settings.google_api_key.get_secret_value())
        else:
            logger.error("Google API Key not found in settings.")
            raise ValueError("Google API Key not configured.")

    else:
        logger.error(f"Unknown or unsupported LLM name: {llm_name}")
        raise ValueError(f"Unknown or unsupported LLM name: {llm_name}")
