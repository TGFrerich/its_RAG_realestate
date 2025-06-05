import logging
from typing import List, Dict, Any
from operator import itemgetter

from langchain_core.documents import Document
from langchain_core.language_models.llms import LLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.vectorstores import VectorStoreRetriever

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def format_docs(docs: List[Document]) -> str:
    """
    Formats a list of retrieved documents into a single string for the prompt context.
    Includes source metadata.

    Args:
        docs (List[Document]): The list of retrieved LangChain Document objects.

    Returns:
        str: A formatted string containing the content and source of each document.
    """
    if not docs:
        return "Kein Kontext gefunden." # No context found

    formatted_strings = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unbekannte Quelle") # Default if source is missing
        content_preview = doc.page_content.replace('\n', ' ').strip()
        formatted_strings.append(f"Dokument {i+1} (Quelle: {source}):\n{content_preview}")

    return "\n\n---\n\n".join(formatted_strings)

def format_notes_input(notes: Dict[str, Any]) -> str:
    """
    Formats the structured meeting notes dictionary into a string for the LLM query.

    Args:
        notes (Dict[str, Any]): The dictionary containing 'general_info' and 'decision_points'.

    Returns:
        str: A formatted string representation of the notes.
    """
    general = notes.get("general_info", {})
    points = notes.get("decision_points", [])

    notes_str = f"**Allgemeine Informationen:**\n"
    notes_str += f"- Thema: {general.get('topic', 'N/A')}\n"
    notes_str += f"- Datum: {general.get('date', 'N/A')}\n"
    notes_str += f"- Teilnehmer: {general.get('attendees', 'N/A')}\n\n"

    if points:
        notes_str += "**Besprochene Punkte & Entscheidungen:**\n"
        for i, point in enumerate(points):
            notes_str += f"{i+1}. Punkt: {point.get('point', 'N/A')}\n"
            notes_str += f"   Entscheidung/Ergebnis: {point.get('decision', 'N/A')}\n"
    else:
        notes_str += "**Keine spezifischen Punkte/Entscheidungen notiert.**\n"

    return notes_str.strip()


def generate_protocol(notes: Dict[str, Any], retriever: VectorStoreRetriever, llm: LLM, prompt_template: ChatPromptTemplate) -> str:
    """
    Generates a protocol draft using the RAG pipeline.

    Args:
        notes (Dict[str, Any]): The structured meeting notes input.
        retriever (VectorStoreRetriever): The initialized vector store retriever.
        llm (LLM): The initialized language model.
        prompt_template (ChatPromptTemplate): The prompt template for the RAG chain.

    Returns:
        str: The generated protocol draft string.

    Raises:
        Exception: For errors during the RAG chain execution.
    """
    logger.info("Starting protocol generation process...")

    print(f"--- Debug: Meeting Notes (received by generate_protocol, type: {type(notes)}) ---")
    print(notes)
    print("--- End Debug: Meeting Notes (received by generate_protocol) ---")

    try:
        # Format the notes into a single query string for retrieval
        # We retrieve based on the combined notes information.
        notes_query = format_notes_input(notes)
        logger.debug(f"Formatted notes for retrieval query: {notes_query}")

        # Define the RAG chain using LCEL
        rag_chain = (
            # Pass the formatted notes through for the prompt, and use it for retrieval
            {"context": itemgetter("notes_query")
                        | retriever
                        | RunnableLambda(
                            lambda docs: (
                                print(f"--- Debug: Retrieved Context (type: {type(docs)}) ---"),
                                print(docs),
                                print("--- End Debug: Retrieved Context ---"),
                                docs
                            )[-1]  # Pass docs through after printing
                          )
                        | format_docs,
             "notes": itemgetter("notes_formatted")}
            | prompt_template
            | RunnableLambda(
                lambda final_prompt: (
                    print(f"--- Debug: Prompt Template Output (type: {type(final_prompt)}) ---"),
                    print(final_prompt),
                    print("--- End Debug: Prompt Template Output ---"),
                    final_prompt
                )[-1]  # Pass through after printing
            )
            | llm
            | StrOutputParser()
        )

        logger.info("Invoking RAG chain...")
        # Prepare the input dictionary for the chain
        chain_input = {
            "notes_query": notes_query,
            "notes_formatted": format_notes_input(notes) # Pass formatted notes again for the prompt
        }
        result = rag_chain.invoke(chain_input)
        logger.info("Protocol generation completed successfully.")
        return result

    except Exception as e:
        logger.error(f"Error during protocol generation: {e}", exc_info=True)
        # Re-raise the exception to be handled by the caller (e.g., Streamlit app)
        raise

import re

def clean_citations(text: str) -> str:
    """
    Removes citation patterns like (Quelle: filename.pdf) from the text using regex.

    Args:
        text (str): The text containing citations.

    Returns:
        str: The text with citation patterns removed.
    """
    # Regex to find "(Quelle: ...)" patterns, accounting for potential variations
    # It looks for "(Quelle:", followed by any characters non-greedily until the closing parenthesis ")"
    # It also handles potential whitespace around the pattern.
    citation_pattern = r'\s*\(Quelle:[^)]+\)\s*'
    cleaned_text = re.sub(citation_pattern, ' ', text).strip() # Replace with space, then strip leading/trailing spaces
    # Replace multiple spaces resulting from removal with a single space
    cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text)
    logger.info("Removed citation patterns from the generated text.")
    return cleaned_text
# Example Usage (for testing purposes, can be removed later)
# if __name__ == '__main__':
#     # This requires setting up dummy retriever, llm, prompt etc.
#     # Example:
#     # from rag_app.retrieval.retriever import initialize_embedding_model, initialize_vector_store, get_retriever
#     # from rag_app.llm.generator import initialize_llm, rag_prompt
#     # from dotenv import load_dotenv
#
#     # load_dotenv()
#     # embed_model = initialize_embedding_model()
#     # vector_store = initialize_vector_store(embed_model)
#     # # Assume vector_store has some data added previously
#     # retriever = get_retriever(vector_store)
#     # llm = initialize_llm()
#
#     # dummy_notes = {
#     #     "general_info": {"topic": "Test Meeting", "date": "2024-01-01", "attendees": "Alice, Bob"},
#     #     "decision_points": [{"point": "Budget Approval", "decision": "Approved 5000 EUR"}]
#     # }
#
#     # try:
#     #     protocol = generate_protocol(dummy_notes, retriever, llm, rag_prompt)
#     #     print("\n--- Generated Protocol ---")
#     #     print(protocol)
#     # except Exception as e:
#     #     print(f"Failed to generate protocol: {e}")