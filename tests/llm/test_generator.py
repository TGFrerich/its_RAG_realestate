import pytest
import os
from unittest.mock import patch, MagicMock

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.llms import LLM
from langchain_mistralai.chat_models import ChatMistralAI

from rag_app.llm.generator import initialize_mistral_llm, rag_prompt

# Define a dummy API key for tests
DUMMY_API_KEY = "test_mistral_api_key"

@pytest.fixture(autouse=True)
def clear_env_vars():
    """Clears relevant environment variables before each test."""
    original_mistral_key = os.environ.pop("MISTRAL_API_KEY", None)
    yield
    if original_mistral_key is not None:
        os.environ["MISTRAL_API_KEY"] = original_mistral_key

class TestInitializeMistralLLM:
    """Tests for the initialize_mistral_llm function."""

    @patch.dict(os.environ, {"MISTRAL_API_KEY": DUMMY_API_KEY})
    @patch("rag_app.llm.generator.ChatMistralAI")
    def test_initialize_mistral_llm_success_with_env_key(self, mock_chat_mistral_ai: MagicMock) -> None:
        """
        Tests successful initialization of ChatMistralAI using API key from environment variable.
        """
        mock_llm_instance = MagicMock(spec=ChatMistralAI)
        mock_chat_mistral_ai.return_value = mock_llm_instance

        llm = initialize_mistral_llm(model_name="test-model", temperature=0.5)

        assert isinstance(llm, LLM)
        mock_chat_mistral_ai.assert_called_once_with(
            model="test-model",
            mistral_api_key=DUMMY_API_KEY,
            temperature=0.5
        )
        assert llm == mock_llm_instance

    @patch("rag_app.llm.generator.ChatMistralAI")
    def test_initialize_mistral_llm_success_with_direct_key(self, mock_chat_mistral_ai: MagicMock) -> None:
        """
        Tests successful initialization of ChatMistralAI using a directly provided API key.
        """
        mock_llm_instance = MagicMock(spec=ChatMistralAI)
        mock_chat_mistral_ai.return_value = mock_llm_instance

        llm = initialize_mistral_llm(api_key=DUMMY_API_KEY, model_name="another-model")

        assert isinstance(llm, LLM)
        mock_chat_mistral_ai.assert_called_once_with(
            model="another-model",
            mistral_api_key=DUMMY_API_KEY
        )
        assert llm == mock_llm_instance

    @patch.dict(os.environ, clear=True) # Ensure MISTRAL_API_KEY is not set
    def test_initialize_mistral_llm_api_key_not_found(self) -> None:
        """
        Tests that a ValueError is raised if the API key is not found.
        """
        with pytest.raises(ValueError, match="Mistral API key not found"):
            initialize_mistral_llm()

    @patch("rag_app.llm.generator.ChatMistralAI", side_effect=ImportError("langchain-mistralai not installed"))
    def test_initialize_mistral_llm_import_error(self, mock_chat_mistral_ai_import: MagicMock) -> None:
        """
        Tests that an ImportError is raised if langchain-mistralai is not installed.
        """
        with pytest.raises(ImportError, match="langchain-mistralai not installed"):
            initialize_mistral_llm(api_key=DUMMY_API_KEY)

    @patch("rag_app.llm.generator.ChatMistralAI", side_effect=Exception("Some API error"))
    def test_initialize_mistral_llm_other_exception(self, mock_chat_mistral_ai_exception: MagicMock) -> None:
        """
        Tests that other exceptions during ChatMistralAI initialization are propagated.
        """
        with pytest.raises(Exception, match="Some API error"):
            initialize_mistral_llm(api_key=DUMMY_API_KEY)

    @patch.dict(os.environ, {"MISTRAL_API_KEY": DUMMY_API_KEY})
    @patch("rag_app.llm.generator.ChatMistralAI")
    def test_initialize_mistral_llm_default_model_name(self, mock_chat_mistral_ai: MagicMock) -> None:
        """
        Tests that the default model name is used if none is provided.
        """
        initialize_mistral_llm()
        mock_chat_mistral_ai.assert_called_once_with(
            model="mistral-large-latest", # Default model
            mistral_api_key=DUMMY_API_KEY
        )

class TestRAGPromptTemplate:
    """Tests for the RAG prompt template."""

    def test_rag_prompt_is_chat_prompt_template(self) -> None:
        """
        Tests that rag_prompt is an instance of ChatPromptTemplate.
        """
        assert isinstance(rag_prompt, ChatPromptTemplate)

    def test_rag_prompt_has_system_and_human_messages(self) -> None:
        """
        Tests that the prompt template contains system and human messages.
        """
        assert len(rag_prompt.messages) == 2
        assert rag_prompt.messages[0].prompt.template.startswith("Du bist ein Assistent") # System
        assert "Bitte erstelle den Protokollentwurf" in rag_prompt.messages[1].prompt.template # Human

    def test_rag_prompt_input_variables(self) -> None:
        """
        Tests that the prompt template has the correct input variables.
        """
        # For from_messages, input variables are derived from templates
        assert "context" in rag_prompt.input_variables
        assert "notes" in rag_prompt.input_variables


# Example of a basic generation call test (mocking the LLM's response)
@patch.dict(os.environ, {"MISTRAL_API_KEY": DUMMY_API_KEY})
@patch("rag_app.llm.generator.ChatMistralAI")
def test_mistral_llm_basic_generation_mocked(mock_chat_mistral_ai_class: MagicMock) -> None:
    """
    Tests a basic generation call by mocking the LLM's invoke method.
    This is a simplified test to ensure the LLM object can be called.
    """
    mock_llm_instance = MagicMock(spec=ChatMistralAI)
    mock_llm_instance.invoke.return_value = "Mocked LLM response"
    mock_chat_mistral_ai_class.return_value = mock_llm_instance

    llm = initialize_mistral_llm(model_name="test-model")
    
    # This is a very basic check; in a real scenario, you'd test the chain
    # using this LLM, but here we just check if invoke can be called.
    response = llm.invoke("Test input")
    
    assert response == "Mocked LLM response"
    mock_llm_instance.invoke.assert_called_once_with("Test input")


@patch.dict(os.environ, {"MISTRAL_API_KEY": DUMMY_API_KEY})
@patch("rag_app.llm.generator.ChatMistralAI")
def test_mistral_llm_generation_api_error(mock_chat_mistral_ai_class: MagicMock) -> None:
    """
    Tests that an exception during the LLM's invoke method (e.g., API error) is propagated.
    """
    mock_llm_instance = MagicMock(spec=ChatMistralAI)
    # Simulate an API error during the invoke call
    mock_llm_instance.invoke.side_effect = Exception("Mistral API Error: Invalid Key")
    mock_chat_mistral_ai_class.return_value = mock_llm_instance

    llm = initialize_mistral_llm(model_name="test-model")
    
    with pytest.raises(Exception, match="Mistral API Error: Invalid Key"):
        llm.invoke("Test input that will trigger the mocked error")
    
    mock_llm_instance.invoke.assert_called_once_with("Test input that will trigger the mocked error")