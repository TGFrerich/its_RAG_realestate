import logging
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def chunk_documents(docs: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Splits a list of LangChain Documents into smaller chunks using RecursiveCharacterTextSplitter.

    Args:
        docs (List[Document]): The list of documents to chunk.
        chunk_size (int): The target size for each chunk (in characters). Defaults to 1000.
        chunk_overlap (int): The overlap between consecutive chunks (in characters). Defaults to 200.

    Returns:
        List[Document]: A list of chunked Document objects. Returns an empty list if the
                        input list is empty.
    """
    if not docs:
        logger.warning("No documents provided for chunking.")
        return []

    logger.info(f"Starting chunking for {len(docs)} documents with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}.")

    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True, # Helps in identifying chunk position within original doc
    )

    try:
        # Split the documents
        chunked_docs = text_splitter.split_documents(docs)
        logger.info(f"Successfully chunked {len(docs)} documents into {len(chunked_docs)} chunks.")
    except Exception as e:
        logger.error(f"Error during document chunking: {e}", exc_info=True)
        return [] # Return empty list on error

    return chunked_docs

# Example Usage (for testing purposes, can be removed later)
# if __name__ == '__main__':
#     # Create some dummy documents
#     dummy_docs = [
#         Document(page_content="This is the first document. It is relatively short.", metadata={"source": "doc1.pdf"}),
#         Document(page_content="This is the second document. It is much longer and will definitely need to be split into multiple chunks to meet the chunk size requirement. We are adding more text here to ensure it exceeds the default chunk size of 1000 characters. " * 20, metadata={"source": "doc2.pdf"}),
#         Document(page_content="A third short document.", metadata={"source": "doc3.pdf"})
#     ]
#
#     chunked_documents = chunk_documents(dummy_docs, chunk_size=200, chunk_overlap=50)
#
#     if chunked_documents:
#         print(f"Chunked into {len(chunked_documents)} documents.")
#         for i, chunk in enumerate(chunked_documents):
#             print(f"--- Chunk {i+1} ---")
#             print(f"Source: {chunk.metadata.get('source', 'N/A')}")
#             print(f"Start Index: {chunk.metadata.get('start_index', 'N/A')}")
#             print(f"Content Preview: {chunk.page_content[:100]}...")
#             print("-" * 15)
#     else:
#         print("Chunking failed or produced no results.")