import logging
import os
from typing import Optional

from langchain_ollama.llms import OllamaLLM
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
        llm = OllamaLLM(
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

from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate

# --- Prompt Template Definition ---

# Define the system and human messages for the prompt template
# Note: Using ChatPromptTemplate structure even with Ollama for potential future compatibility
# with chat models and standard LCEL patterns.
RAG_PROMPT_TEMPLATE = """
**System:**
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

**JSON-Notizen zur Versammlung:**
```json
{notes}
```
**Relevanter Kontext aus Dokumenten::**
```
{context}
```

Human:
Erstelle nun bitte den vollständigen Protokollentwurf für die oben spezifizierte Wohnungseigentümerversammlung gemäß ALLEN Anweisungen.

Protokollentwurf:
"""

# Create the ChatPromptTemplate instance
rag_prompt = ChatPromptTemplate.from_messages([HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=["context", "notes"],template=RAG_PROMPT_TEMPLATE))])

logger.info("RAG prompt template created.")

# --- Add Core RAG Chain function below ---
# --- Add Prompt Engineering and other generator functions below ---