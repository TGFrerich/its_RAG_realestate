import pytest
from pathlib import Path
from rag_app.utils.helpers import list_uploaded_files

def test_list_uploaded_files(tmp_path: Path):
    """
    Tests the list_uploaded_files function.
    """
    # Create some dummy files
    (tmp_path / "doc1.pdf").touch()
    (tmp_path / "doc2.pdf").touch()
    (tmp_path / "other.txt").touch()
    (tmp_path / "subfolder").mkdir()
    (tmp_path / "subfolder" / "doc3.pdf").touch()

    # Test the function
    pdf_files = list_uploaded_files(tmp_path)

    # Assertions
    assert len(pdf_files) == 2
    assert tmp_path / "doc1.pdf" in pdf_files
    assert tmp_path / "doc2.pdf" in pdf_files
    assert tmp_path / "subfolder" / "doc3.pdf" not in pdf_files