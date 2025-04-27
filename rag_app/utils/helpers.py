import os
from pathlib import Path
import logging

# Basic Logging Setup (can be enhanced later)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def list_uploaded_files(directory: str | Path) -> list[Path]:
    """
    Lists all PDF files in the specified directory.

    Args:
        directory (str | Path): The path to the directory containing the documents.

    Returns:
        list[Path]: A list of Path objects for the PDF files found. Returns an empty list
                    if the directory doesn't exist or no PDFs are found.
    """
    doc_path = Path(directory)
    if not doc_path.is_dir():
        logging.warning(f"Document directory not found: {directory}")
        return []

    pdf_files = sorted([f for f in doc_path.glob("*.pdf") if f.is_file()])
    logging.info(f"Found {len(pdf_files)} PDF files in {directory}.")
    return pdf_files

# --- Add other utility functions below as needed ---