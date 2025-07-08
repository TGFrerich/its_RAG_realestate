import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document
from rag_app.core.pipeline import (
    format_docs,
    format_notes_input,
    clean_citations,
    generate_protocol,
)

def test_format_docs():
    """
    Tests the format_docs function.
    """
    # Arrange
    docs = [
        Document(page_content="doc1 content", metadata={"source": "doc1.pdf"}),
        Document(page_content="doc2 content", metadata={"source": "doc2.pdf"}),
    ]

    # Act
    formatted_docs = format_docs(docs)

    # Assert
    assert "Dokument 1 (Quelle: doc1.pdf):" in formatted_docs
    assert "doc1 content" in formatted_docs
    assert "Dokument 2 (Quelle: doc2.pdf):" in formatted_docs
    assert "doc2 content" in formatted_docs

def test_format_notes_input():
    """
    Tests the format_notes_input function.
    """
    # Arrange
    notes = {
        "general_info": {"topic": "Test Meeting", "date": "2024-01-01", "attendees": "Alice, Bob"},
        "decision_points": [{"point": "Budget Approval", "decision": "Approved 5000 EUR"}],
    }

    # Act
    formatted_notes = format_notes_input(notes)

    # Assert
    assert "**Allgemeine Informationen:**" in formatted_notes
    assert "Thema: Test Meeting" in formatted_notes
    assert "**Besprochene Punkte & Entscheidungen:**" in formatted_notes
    assert "1. Punkt: Budget Approval" in formatted_notes

def test_clean_citations():
    """
    Tests the clean_citations function.
    """
    # Arrange
    text = "This is a sentence (Quelle: doc1.pdf) with a citation."

    # Act
    cleaned_text = clean_citations(text)

    # Assert
    assert "(Quelle: doc1.pdf)" not in cleaned_text
    assert "This is a sentence with a citation." in cleaned_text

def test_generate_protocol():
    """
    Tests the generate_protocol function with a mocked RAG chain.
    """
    # Arrange
    mock_retriever = MagicMock()
    mock_llm = MagicMock()
    mock_prompt = MagicMock()
    notes = {"general_info": {}, "decision_points": []}

    # Mock the chain
    mock_chain = MagicMock()
    mock_prompt.pipe.return_value = mock_chain
    mock_chain.pipe.return_value = mock_chain
    mock_chain.invoke.return_value = "Mocked protocol"

    # Act
    with patch("rag_app.core.pipeline.itemgetter") as mock_itemgetter, \
         patch("rag_app.core.pipeline.RunnableLambda") as mock_runnable_lambda, \
         patch("rag_app.core.pipeline.StrOutputParser") as mock_parser:
        
        mock_itemgetter.return_value = MagicMock()
        mock_runnable_lambda.return_value = MagicMock()
        mock_parser.return_value = MagicMock()

        protocol = generate_protocol(notes, mock_retriever, mock_llm, mock_prompt)

    # Assert
    assert protocol == "Mocked protocol"