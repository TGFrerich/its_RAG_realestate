import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from rag_app.retrieval.retriever import (
    initialize_embedding_model,
    initialize_vector_store,
    add_documents_to_db,
    delete_document_from_db,
    get_retriever,
)

@patch("rag_app.retrieval.retriever.HuggingFaceEmbeddings")
def test_initialize_embedding_model(mock_embeddings: MagicMock):
    """
    Tests the initialize_embedding_model function.
    """
    # Arrange
    mock_instance = mock_embeddings.return_value

    # Act
    embedding_model = initialize_embedding_model("test-model")

    # Assert
    assert embedding_model == mock_instance
    mock_embeddings.assert_called_once_with(model_name="test-model", model_kwargs={'trust_remote_code': True})

@patch("rag_app.retrieval.retriever.Chroma")
def test_initialize_vector_store(mock_chroma: MagicMock, tmp_path: Path):
    """
    Tests the initialize_vector_store function.
    """
    # Arrange
    mock_instance = mock_chroma.return_value
    embedding_function = MagicMock()

    # Act
    vector_store = initialize_vector_store(embedding_function, str(tmp_path))

    # Assert
    assert vector_store == mock_instance
    mock_chroma.assert_called_once_with(
        persist_directory=str(tmp_path),
        embedding_function=embedding_function,
    )

def test_add_documents_to_db():
    """
    Tests the add_documents_to_db function.
    """
    # Arrange
    mock_vector_store = MagicMock()
    docs = [Document(page_content="test", metadata={"source": "test.pdf"})]

    # Act
    add_documents_to_db(mock_vector_store, docs)

    # Assert
    mock_vector_store.add_documents.assert_called_once_with(docs)

def test_delete_document_from_db():
    """
    Tests the delete_document_from_db function.
    """
    # Arrange
    mock_vector_store = MagicMock()
    mock_vector_store.get.return_value = {"ids": ["1", "2"]}
    filename = "test.pdf"

    # Act
    delete_document_from_db(mock_vector_store, filename)

    # Assert
    mock_vector_store.get.assert_called_once_with(where={"source": filename}, include=[])
    mock_vector_store.delete.assert_called_once_with(ids=["1", "2"])

def test_get_retriever():
    """
    Tests the get_retriever function.
    """
    # Arrange
    mock_vector_store = MagicMock()
    mock_retriever = MagicMock()
    mock_vector_store.as_retriever.return_value = mock_retriever

    # Act
    retriever = get_retriever(mock_vector_store, k=5)

    # Assert
    assert retriever == mock_retriever
    mock_vector_store.as_retriever.assert_called_once_with(
        search_type="similarity",
        search_kwargs={"k": 5},
    )