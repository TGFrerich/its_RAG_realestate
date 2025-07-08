import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from rag_app.doc_processing.loader import load_pdf_documents

# Create a dummy PDF for testing
@pytest.fixture
def dummy_pdf(tmp_path: Path) -> Path:
    """
    Creates a dummy PDF file for testing.
    """
    pdf_path = tmp_path / "dummy.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /Resources <<>> /MediaBox [0 0 612 792] /Contents 4 0 R >>\nendobj\n4 0 obj\n<< /Length 55 >>\nstream\nBT\n/F1 12 Tf\n100 100 Td\n(Hello World) Tj\nET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000010 00000 n \n0000000079 00000 n \n0000000173 00000 n \n0000000293 00000 n \ntrailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n409\n%%EOF")
    return pdf_path

@patch("rag_app.doc_processing.loader.PDFPlumberLoader")
def test_load_pdf_documents(mock_loader: MagicMock, dummy_pdf: Path):
    """
    Tests the load_pdf_documents function with a mock PDF.
    """
    # Arrange
    mock_instance = mock_loader.return_value
    mock_instance.load.return_value = [Document(page_content="Hello World", metadata={"source": "dummy.pdf"})]

    # Act
    loaded_docs = load_pdf_documents([dummy_pdf])

    # Assert
    assert len(loaded_docs) == 1
    assert loaded_docs[0].page_content == "Hello World"
    assert loaded_docs[0].metadata["source"] == "dummy.pdf"
    mock_loader.assert_called_once_with(str(dummy_pdf), extract_images=False)
    mock_instance.load.assert_called_once()

def test_load_pdf_documents_non_existent_file():
    """
    Tests that load_pdf_documents handles non-existent files gracefully.
    """
    non_existent_file = Path("non_existent.pdf")
    loaded_docs = load_pdf_documents([non_existent_file])
    assert len(loaded_docs) == 0

def test_load_pdf_documents_not_a_pdf():
    """
    Tests that load_pdf_documents handles non-pdf files gracefully.
    """
    not_a_pdf = Path("not_a_pdf.txt")
    not_a_pdf.touch()
    loaded_docs = load_pdf_documents([not_a_pdf])
    assert len(loaded_docs) == 0