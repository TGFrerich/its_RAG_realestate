import logging
import os
from typing import Optional

from dotenv import load_dotenv
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.language_models.llms import LLM
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def initialize_mistral_llm(
    model_name: str = "mistral-large-latest",
    api_key: Optional[str] = None,
    **kwargs,
) -> LLM:
    """
    Initializes and returns a Mistral AI chat model instance.

    Args:
        model_name (str): The name of the Mistral model to use.
                          Defaults to "mistral-large-latest".
        api_key (Optional[str]): The Mistral API key. If None, it will try to load
                                 from the MISTRAL_API_KEY environment variable.
        **kwargs: Additional keyword arguments to pass to ChatMistralAI (e.g., temperature).

    Returns:
        LLM: An instance of the ChatMistralAI model.

    Raises:
        ValueError: If the API key is not provided and cannot be found in env vars.
        ImportError: If langchain-mistralai is not installed.
        Exception: For other potential errors during LLM initialization.
    """
    if api_key is None:
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            logger.error("Mistral API key not found. Please set MISTRAL_API_KEY environment variable.")
            raise ValueError("Mistral API key not found. Set MISTRAL_API_KEY environment variable.")
        logger.info("Loaded Mistral API key from environment variable.")

    try:
        logger.info(f"Initializing ChatMistralAI with model='{model_name}'.")
        llm = ChatMistralAI(
            model=model_name,
            mistral_api_key=api_key,
            **kwargs
        )
        # You might want to add a simple invocation test here if needed,
        # similar to the commented-out Ollama one, but be mindful of API costs.
        # e.g., llm.invoke("Hi")
        logger.info("ChatMistralAI LLM initialized successfully.")
        return llm
    except ImportError:
        logger.error("langchain-mistralai package not found. Please install it: pip install langchain-mistralai")
        raise
    except Exception as e:
        logger.error(f"Failed to initialize ChatMistralAI '{model_name}': {e}", exc_info=True)
        raise

# --- Prompt Template Definition ---

SYSTEM_MESSAGE_CONTENT = """Du bist ein Assistent zur Erstellung von Protokollen für Immobiliengespräche auf Deutsch.
Deine Aufgabe ist es, basierend auf den untenstehenden Notizen und dem bereitgestellten Kontext aus relevanten Dokumenten einen Protokollentwurf zu erstellen.
Halte dich strikt an die folgenden Anweisungen:
1.  **Sprache:** Das gesamte Protokoll muss auf **Deutsch** verfasst sein.
2.  **Kontextnutzung:** Verwende **ausschließlich** die Informationen aus dem Abschnitt "Kontext". Füge keine Informationen hinzu, die nicht im Kontext enthalten sind. Wenn der Kontext keine relevanten Informationen für einen Notizpunkt enthält, gib dies explizit an (z.B. "Keine relevanten Informationen im Kontext gefunden.").
3.  **Struktur:** Orientiere dich an der Struktur der "Notizen", um das Protokoll zu gliedern.
4.  **Zitierung:** Wenn du Informationen aus dem Kontext verwendest, **musst** du die Quelle **direkt nach der Information** im folgenden Format zitieren: `(Quelle: dateiname.pdf)`. Der Dateiname befindet sich in den Metadaten des Kontexts.
5.  **Formulierung:** Formuliere klare und prägnise Protokollsätze."""

HUMAN_MESSAGE_CONTENT_TEMPLATE = """**Kontext:**
{context}

**Notizen:**
{notes}

Bitte erstelle den Protokollentwurf basierend auf den obigen Notizen und dem Kontext. Beachte alle Anweisungen, insbesondere die Zitierpflicht und die ausschließliche Nutzung des Kontexts.

**Protokollentwurf:**"""

# Create the ChatPromptTemplate instance
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_MESSAGE_CONTENT),
    ("human", HUMAN_MESSAGE_CONTENT_TEMPLATE)
])

logger.info("RAG prompt template created using from_messages.")

# --- Add Core RAG Chain function below ---
# --- Add Prompt Engineering and other generator functions below ---
# Note: The initialize_llm function for Ollama has been replaced by initialize_mistral_llm.
# If Ollama support is still needed, the old function can be uncommented and renamed.