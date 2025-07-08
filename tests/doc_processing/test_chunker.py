import pytest
from langchain_core.documents import Document
from rag_app.doc_processing.chunker import chunk_documents

def test_chunk_documents():
    """
    Tests the chunk_documents function with a sample document.
    """
    # Arrange
    docs = [Document(page_content="This is a test document that is long enough to be chunked.", metadata={"source": "doc1.pdf"})]

    # Act
    chunked_docs = chunk_documents(docs, chunk_size=20, chunk_overlap=5)

    # Assert
    assert len(chunked_docs) > 1
    for doc in chunked_docs:
        assert "source" in doc.metadata
        assert doc.metadata["source"] == "doc1.pdf"

def test_chunk_documents_empty_list():
    """
    Tests that chunk_documents handles an empty list of documents.
    """
    # Arrange
    docs = []

    # Act
    chunked_docs = chunk_documents(docs)

    # Assert
    assert len(chunked_docs) == 0

def test_chunk_documents_small_document():
    """
    Tests that a document smaller than the chunk size is not chunked.
    """
    # Arrange
    docs = [Document(page_content="This is a short document.", metadata={"source": "doc2.pdf"})]

    # Act
    chunked_docs = chunk_documents(docs, chunk_size=100, chunk_overlap=10)

    # Assert
    assert len(chunked_docs) == 1
    assert chunked_docs[0].page_content == "This is a short document."