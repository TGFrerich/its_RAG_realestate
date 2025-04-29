import logging
import os
from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2" # A sensible default

def initialize_embedding_model(model_name: Optional[str] = None) -> Embeddings:
    """
    Initializes and returns a HuggingFace embedding model.

    Args:
        model_name (Optional[str]): The name of the Hugging Face model to use for embeddings.
                                    If None, uses the value from the EMBEDDING_MODEL_NAME
                                    environment variable or falls back to DEFAULT_EMBED_MODEL.

    Returns:
        Embeddings: An instance of HuggingFaceEmbeddings.

    Raises:
        ValueError: If the model name is not specified and cannot be found in env vars.
        ImportError: If langchain-huggingface is not installed.
        Exception: For other potential errors during model initialization.
    """
    if model_name is None:
        model_name = os.getenv("EMBEDDING_MODEL_NAME", DEFAULT_EMBED_MODEL)
        logger.info(f"Model name not provided, using from env/default: '{model_name}'")

    if not model_name:
         raise ValueError("Embedding model name must be provided either as an argument or via EMBEDDING_MODEL_NAME environment variable.")

    try:
        logger.info(f"Initializing HuggingFace embedding model: {model_name}")
        # Specify device explicitly if needed, e.g., model_kwargs={'device': 'cuda'} or 'cpu'
        # By default, sentence-transformers tries to use GPU if available.
        embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'trust_remote_code': True} # Adjust if needed based on model
            # cache_folder= # Optional: specify a cache directory
        )
        logger.info("HuggingFace embedding model initialized successfully.")
        return embedding_model
    except ImportError:
        logger.error("langchain-huggingface package not found. Please install it: pip install langchain-huggingface")
        raise
    except Exception as e:
        logger.error(f"Failed to initialize embedding model '{model_name}': {e}", exc_info=True)
        raise

from pathlib import Path
from typing import List
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

DEFAULT_VECTOR_DB_DIR = "./vector_db"

def initialize_vector_store(embedding_function: Embeddings, persist_directory: Optional[str] = None) -> Chroma:
    """
    Initializes or loads a Chroma vector store.

    Args:
        embedding_function (Embeddings): The embedding function to use.
        persist_directory (Optional[str]): The directory to persist the vector store.
                                           If None, uses the value from the VECTOR_DB_DIR
                                           environment variable or falls back to DEFAULT_VECTOR_DB_DIR.

    Returns:
        Chroma: An initialized Chroma vector store instance.

    Raises:
        ValueError: If the persist directory is not specified and cannot be found in env vars.
        ImportError: If chromadb or langchain-community is not installed.
        Exception: For other potential errors during initialization.
    """
    if persist_directory is None:
        persist_directory = os.getenv("VECTOR_DB_DIR", DEFAULT_VECTOR_DB_DIR)
        logger.info(f"Persist directory not provided, using from env/default: '{persist_directory}'")

    if not persist_directory:
        raise ValueError("Vector store persist directory must be provided either as an argument or via VECTOR_DB_DIR environment variable.")

    persist_path = Path(persist_directory)
    persist_path.mkdir(parents=True, exist_ok=True) # Ensure directory exists

    logger.info(f"Initializing Chroma vector store with persistence directory: {persist_path}")

    try:
        vector_store = Chroma(
            persist_directory=str(persist_path),
            embedding_function=embedding_function
            # collection_name= # Optional: specify a collection name if needed
        )
        logger.info("Chroma vector store initialized/loaded successfully.")
        return vector_store
    except ImportError:
        logger.error("chromadb or langchain-community package not found. Please install them: pip install chromadb langchain-community")
        raise
    except Exception as e:
        logger.error(f"Failed to initialize Chroma vector store at '{persist_path}': {e}", exc_info=True)
        raise

def add_documents_to_db(vector_store: Chroma, docs: List[Document]) -> None:
    """
    Adds a list of documents to the Chroma vector store.

    Ensures 'source' metadata is present in the documents before adding.

    Args:
        vector_store (Chroma): The initialized Chroma vector store instance.
        docs (List[Document]): The list of LangChain Document objects to add.

    Raises:
        ValueError: If input `docs` list is empty.
        Exception: For errors during the adding process.
    """
    if not docs:
        logger.warning("No documents provided to add to the vector store.")
        raise ValueError("Input document list cannot be empty.")

    # Ensure source metadata is present (important for deletion)
    for doc in docs:
        if "source" not in doc.metadata or not doc.metadata["source"]:
             logger.warning(f"Document missing 'source' metadata: {doc.page_content[:50]}...")
             # Decide on handling: raise error, assign default, or skip? For now, log warning.
             # raise ValueError("All documents must have 'source' metadata (e.g., filename).")

    doc_sources = list(set(d.metadata.get("source", "Unknown") for d in docs))
    logger.info(f"Adding {len(docs)} document chunks to vector store from sources: {doc_sources}")

    try:
        # Chroma's add_documents handles batching internally to some extent
        vector_store.add_documents(docs)
        # Persist changes explicitly if needed (Chroma often persists automatically with directory)
        # vector_store.persist() # Check Chroma documentation if explicit persist is required
        logger.info(f"Successfully added {len(docs)} chunks to the vector store.")
    except Exception as e:
        logger.error(f"Failed to add documents to vector store: {e}", exc_info=True)
        raise

def delete_document_from_db(vector_store: Chroma, filename: str) -> None:
    """
    Deletes all document chunks associated with a specific source filename
    from the Chroma vector store using metadata filtering.

    Args:
        vector_store (Chroma): The initialized Chroma vector store instance.
        filename (str): The source filename to filter and delete documents for.

    Raises:
        ValueError: If filename is empty.
        Exception: For errors during the deletion process.
    """
    if not filename:
        logger.error("Filename cannot be empty for deletion.")
        raise ValueError("Filename must be provided for deletion.")

    logger.info(f"Attempting to delete document chunks with source: {filename}")

    try:
        # 1. Find IDs of documents matching the source filename
        # Note: Chroma's API for direct ID retrieval based on metadata might vary.
        # A common pattern is to get all IDs and then filter, or use `get` with where filter.
        results = vector_store.get(where={"source": filename}, include=[]) # include=[] fetches only IDs

        if not results or not results.get("ids"):
            logger.warning(f"No document chunks found with source '{filename}' to delete.")
            return

        ids_to_delete = results["ids"]
        logger.info(f"Found {len(ids_to_delete)} chunk(s) with source '{filename}'. Deleting...")

        # 2. Delete the documents by their IDs
        vector_store.delete(ids=ids_to_delete)
        # vector_store.persist() # Check if explicit persist is needed after deletion

        logger.info(f"Successfully deleted {len(ids_to_delete)} chunk(s) for source: {filename}")

    except Exception as e:
        logger.error(f"Failed to delete document chunks for source '{filename}': {e}", exc_info=True)
        raise

from langchain_core.vectorstores import VectorStoreRetriever

DEFAULT_RETRIEVER_K = 4

def get_retriever(vector_store: VectorStore, search_type: str = "similarity", k: Optional[int] = None) -> VectorStoreRetriever:
    """
    Creates and returns a retriever from the given vector store.

    Args:
        vector_store (VectorStore): The initialized vector store instance (e.g., Chroma).
        search_type (str): The type of search to perform (e.g., "similarity", "mmr").
                           Defaults to "similarity".
        k (Optional[int]): The number of documents to retrieve. If None, uses the value
                           from the RETRIEVER_K environment variable or falls back to
                           DEFAULT_RETRIEVER_K.

    Returns:
        VectorStoreRetriever: A retriever instance configured for the vector store.

    Raises:
        ValueError: If the vector store is not provided.
        Exception: For other potential errors during retriever creation.
    """
    if not vector_store:
        raise ValueError("Vector store must be provided to create a retriever.")

    if k is None:
        try:
            k = int(os.getenv("RETRIEVER_K", DEFAULT_RETRIEVER_K))
        except ValueError:
            logger.warning(f"Invalid RETRIEVER_K env var. Using default: {DEFAULT_RETRIEVER_K}")
            k = DEFAULT_RETRIEVER_K
        logger.info(f"Retriever 'k' not provided, using from env/default: {k}")

    search_kwargs = {'k': k}
    logger.info(f"Creating retriever with search_type='{search_type}' and k={k}")

    try:
        retriever = vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
        logger.info("Retriever created successfully.")
        return retriever
    except Exception as e:
        logger.error(f"Failed to create retriever: {e}", exc_info=True)
        raise
# --- Add retriever function below ---
# --- Add other retrieval functions below (vector store, retriever) ---