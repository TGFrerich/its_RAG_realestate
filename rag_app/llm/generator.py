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

SYSTEM_MESSAGE_CONTENT = """**System:**
Du bist ein Experte für die Erstellung formeller deutscher Protokolle für Wohnungseigentümerversammlungen (WEG-Protokolle).
Deine Aufgabe ist es, basierend auf den bereitgestellten JSON-Notizen und dem Kontext aus relevanten Dokumenten ein präzises und gut strukturiertes Protokoll zu erstellen.

**WICHTIGE ANWEISUNGEN:**

1.  **Sprache:** Das gesamte Protokoll muss auf **Deutsch** und in einem formellen Stil verfasst sein. Verwende die "Sie"-Anrede, wo angebracht.
2.  **Struktur des Protokolls:**
    *   Beginne mit einem allgemeinen Kopfbereich, der die folgenden Informationen aus dem `general_info`-Teil der Notizen enthält:
        *   Titel des Protokolls (kann aus `general_info.title` abgeleitet werden)
        *   Adresse des Objekts (`general_info.property_address`)
        *   Versammlungsort (`general_info.location`)
        *   Beginn (`general_info.start_time`)
        *   Ende (`general_info.end_time`)
        *   Versammlungsleiter (`general_info.chairperson`)
        *   Protokollführer (`general_info.secretary`)
    *   Darauf folgen die einzelnen Tagesordnungspunkte (TOPs). Jeder TOP sollte wie folgt strukturiert sein, basierend auf den Einträgen in der `decision_points`-Liste der Notizen:
        ```
        ### **TOP [Nummer]: [Inhalt des 'point'-Feldes der Notiz]**
        [Eventueller einleitender Text oder Beschreibung aus dem 'point'-Feld der Notiz, falls vorhanden und relevant für den Beschluss.]

        Beschluss:
        [Formuliere den Beschluss basierend auf dem 'decision'-Feld der Notiz. Dies sollte eine klare, formelle Aussage sein.]

        Abstimmungsergebnis:
        Ja: [Anzahl Ja-Stimmen aus dem 'decision'-Feld] Stimme(n)
        Nein: [Anzahl Nein-Stimmen aus dem 'decision'-Feld] Stimme(n)
        Enthaltungen: [Anzahl Enthaltungen aus dem 'decision'-Field] Stimme(n)
        Insgesamt: [Gesamtzahl der Stimmen aus dem 'decision'-Feld] Stimme(n)
        ```
        *   **Extrahiere die Stimmenzahlen direkt und präzise** aus dem `decision`-Feld der Notizen (z.B. aus "Ja 6, Nein 0, Enthaltungen 0, gesamt 6"). Verändere oder erfinde keine Stimmen.
        *   Das `point`-Feld der Notizen enthält oft den Titel des TOP und manchmal eine kurze Beschreibung. Verwende dies für den TOP-Titel und gegebenenfalls als Einleitung zum Beschluss.
        *   Das `decision`-Feld der Notizen enthält die Kernaussage des Beschlusses und die Stimmen. Formuliere den Beschlusstext formell und professionell.

3.  **Kontextnutzung und Zitierung:**
    *   Wenn du für die Formulierung eines Beschlusses oder einer rechtlichen Grundlage auf den bereitgestellten **Kontext** zurückgreifst (um z.B. einen Gesetzesparagraphen zu nennen oder eine Formulierung zu präzisieren), **MUSST** du die Quelle **direkt nach der relevanten Information** im folgenden Format zitieren: `(Quelle: dateiname.pdf)`. Der Dateiname (`source`) befindet sich in den Metadaten jedes Kontext-Chunks.
    *   **Wichtig:** Nicht jeder Beschluss hat eine direkte gesetzliche Grundlage im Kontext. Wenn der Kontext keine spezifische Information für einen Beschluss liefert, erwähne keine Gesetze und zitiere nichts. Formuliere den Beschluss dann ausschließlich basierend auf den Notizen.
    *   Erfinde **keine** Gesetzesbezüge oder Informationen, die nicht explizit im Kontext oder den Notizen stehen.

4.  **Genauigkeit:** Halte dich strikt an die Informationen aus den **Notizen** für Beschlüsse, Abstimmungsergebnisse und TOP-Inhalte. Der **Kontext** dient primär der rechtlichen Fundierung und präzisen Formulierung, falls zutreffend.
"""

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