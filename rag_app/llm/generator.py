import logging
import os
from typing import Optional

from langchain_community.llms import Ollama
from langchain_core.language_models.llms import LLM

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_MODEL = "mistral" # Default model if not specified

def initialize_llm(model_name: Optional[str] = None, base_url: Optional[str] = None) -> LLM:
    """
    Initializes and returns an Ollama language model instance.

    Args:
        model_name (Optional[str]): The name of the Ollama model to use (e.g., 'mistral', 'llama3').
                                    If None, uses the value from the OLLAMA_MODEL environment
                                    variable or falls back to DEFAULT_OLLAMA_MODEL.
        base_url (Optional[str]): The base URL for the Ollama API. If None, uses the value
                                  from the OLLAMA_BASE_URL environment variable or the
                                  default Ollama URL (http://localhost:11434).

    Returns:
        LLM: An instance of the Ollama language model.

    Raises:
        ValueError: If the model name is not specified and cannot be found in env vars.
        ImportError: If langchain-community is not installed.
        Exception: For other potential errors during LLM initialization (e.g., connection issues).
    """
    if model_name is None:
        model_name = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
        logger.info(f"Ollama model name not provided, using from env/default: '{model_name}'")

    if not model_name:
        raise ValueError("Ollama model name must be provided either as an argument or via OLLAMA_MODEL environment variable.")

    # Get base URL from env var if not provided
    if base_url is None:
        base_url = os.getenv("OLLAMA_BASE_URL") # Will be None if not set
        if base_url:
            logger.info(f"Using Ollama base URL from environment variable: {base_url}")
        # If still None, Ollama class uses its default http://localhost:11434

    try:
        logger.info(f"Initializing Ollama LLM with model='{model_name}' and base_url='{base_url or 'default'}'.")
        llm = Ollama(
            model=model_name,
            base_url=base_url # Pass None if not specified, Ollama handles default
            # Other parameters like temperature, top_k, etc., can be added here if needed
            # temperature=0.7
        )
        # Optional: Add a simple check to see if the Ollama server is reachable
        # try:
        #     llm.invoke("Hi") # Simple invocation to test connection
        #     logger.info("Ollama LLM connection test successful.")
        # except Exception as conn_err:
        #     logger.warning(f"Could not connect to Ollama server at {base_url or 'default'} with model {model_name}. Ensure Ollama is running. Error: {conn_err}")
        #     # Depending on requirements, you might want to raise an error here

        logger.info("Ollama LLM initialized successfully.")
        return llm
    except ImportError:
        logger.error("langchain-community package not found. Please install it: pip install langchain-community")
        raise
    except Exception as e:
        logger.error(f"Failed to initialize Ollama LLM '{model_name}': {e}", exc_info=True)
        raise

from langchain_core.prompts import ChatPromptTemplate

# --- Prompt Template Definition ---

# Define the system and human messages for the prompt template
# Note: Using ChatPromptTemplate structure even with Ollama for potential future compatibility
# with chat models and standard LCEL patterns.
RAG_PROMPT_TEMPLATE = """
**System:**
Du bist ein Assistent zur Erstellung von Protokollen für Immobiliengespräche auf Deutsch.
Deine Aufgabe ist es, basierend auf den untenstehenden Notizen und dem bereitgestellten Kontext aus relevanten Dokumenten einen Protokollentwurf zu erstellen.
Halte dich strikt an die folgenden Anweisungen:
1.  **Sprache:** Das gesamte Protokoll muss auf **Deutsch** verfasst sein.
2.  **Kontextnutzung:** Verwende **ausschließlich** die Informationen aus dem Abschnitt "Kontext". Füge keine Informationen hinzu, die nicht im Kontext enthalten sind. Wenn der Kontext keine relevanten Informationen für einen Notizpunkt enthält, gib dies explizit an (z.B. "Keine relevanten Informationen im Kontext gefunden.").
3.  **Struktur:** Orientiere dich an der Struktur der "Notizen", um das Protokoll zu gliedern.
4.  **Zitierung:** Wenn du Informationen aus dem Kontext verwendest, **musst** du die Quelle **direkt nach der Information** im folgenden Format zitieren: `(Quelle: dateiname.pdf)`. Der Dateiname befindet sich in den Metadaten des Kontexts.
5.  **Formulierung:** Formuliere klare und prägnise Protokollsätze.

**Kontext:**
{context}

**Notizen:**
{notes}

**Human:**
Bitte erstelle den Protokollentwurf basierend auf den obigen Notizen und dem Kontext. Beachte alle Anweisungen, insbesondere die Zitierpflicht und die ausschließliche Nutzung des Kontexts.

**Protokollentwurf:**
"""

# Create the ChatPromptTemplate instance
rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

logger.info("RAG prompt template created.")

# --- Add Core RAG Chain function below ---
# --- Add Prompt Engineering and other generator functions below ---