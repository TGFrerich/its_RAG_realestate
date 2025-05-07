import logging
from pathlib import Path
from typing import List
from langchain_community.document_loaders.pdf import PDFPlumberLoader
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_pdf_documents(file_paths: List[Path]) -> List[Document]:
    """
    Loads content from a list of PDF files using PyPDFLoader.

    Args:
        file_paths (List[Path]): A list of Path objects pointing to the PDF files.

    Returns:
        List[Document]: A list of LangChain Document objects, where each Document
                        represents a loaded PDF. Returns an empty list if input is empty
                        or errors occur during loading.
    """
    if not file_paths:
        logger.warning("No file paths provided for loading.")
        return []

    all_docs: List[Document] = []
    for file_path in file_paths:
        if not file_path.is_file() or file_path.suffix.lower() != ".pdf":
            logger.warning(f"Skipping invalid or non-PDF file: {file_path}")
            continue

        try:
            loader = PDFPlumberLoader(str(file_path), extract_images=False) # extract_images=False for efficiency
            docs = loader.load() # load() returns a list of Documents (often one per page)
            logger.info(f"Loaded {len(docs)} pages/documents from {file_path.name}")
            # Add source metadata to each document/page
            for doc in docs:
                doc.metadata["source"] = file_path.name
            all_docs.extend(docs)
        except Exception as e:
            logger.error(f"Error loading PDF file {file_path.name}: {e}", exc_info=True)
            # Optionally, decide whether to continue with other files or raise exception

    logger.info(f"Successfully loaded a total of {len(all_docs)} pages from {len(file_paths)} PDF file(s).")
    return all_docs

# Example Usage (for testing purposes, can be removed later)
# if __name__ == '__main__':
#     # Create a dummy PDF in a 'temp_docs' folder for testing
#     temp_dir = Path("./temp_docs")
#     temp_dir.mkdir(exist_ok=True)
#     dummy_pdf_path = temp_dir / "dummy.pdf"
#     # You'd need a library like reportlab to actually create a PDF here
#     # For now, assume dummy.pdf exists or test with real PDFs
#     if dummy_pdf_path.exists():
#         loaded_documents = load_pdf_documents([dummy_pdf_path])
#         if loaded_documents:
#             print(f"Loaded {len(loaded_documents)} documents.")
#             # print(loaded_documents[0].page_content[:500]) # Print first 500 chars of first doc
#             print(loaded_documents[0].metadata)
#     else:
#         print(f"Please create a dummy PDF at {dummy_pdf_path} for testing.")