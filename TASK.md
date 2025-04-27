# Project Tasks: its_RAG_realestate - Initial Prototype

This task list outlines the initial steps to build the local prototype. Tasks should be checked off as completed.

## Phase 1: Setup & Basic UI

-   [ ] **Environment Setup:**
    -   [x] Create project directory structure (`its_rag_realestate/`, `rag_app/`, etc.).
    -   [x] Set up Python virtual environment.
    -   [x] Install initial dependencies from `requirements.txt` (`streamlit`, `langchain`, `langchain-community`, `langchain-huggingface`, `sentence-transformers`, `pypdf`, `ollama`, `chromadb`, `python-dotenv`, `PyYAML`).
    -   [x] Set up Ollama and pull a suitable LLM (e.g., `mistral` or `llama3`). Verify Ollama server is running.
    -   [x] Create and populate `.env` file with initial configuration (paths, model names, chunk settings).
    -   [x] Initialize `.gitignore`.
    -   [x] Create basic `README.md`.
-   [ ] **Basic Streamlit App (`app.py`):**
    -   [x] Set up basic Streamlit page layout (`st.set_page_config`, title).
    -   [x] Load configuration from `.env`.
    -   [x] Implement sidebar structure for Document Management.
    -   [x] Implement main area structure (columns for Input/Output).
    -   [x] Initialize basic session state variables (`protocol_draft`, `meeting_notes`, etc.).

## Phase 2: Document Management & Processing

-   [ ] **Document Upload & Storage (`app.py`, `rag_app/utils/helpers.py`):**
    -   [ ] Implement file upload (`st.file_uploader`) in the sidebar.
    -   [ ] Save uploaded PDF files to the `documents/` directory.
    -   [ ] Implement function to list files in `documents/` directory (`list_uploaded_files`).
    -   [ ] Display the list of uploaded files in the sidebar.
    -   [ ] Implement file download (`st.download_button`) for listed files.
-   [ ] **Document Loading (`rag_app/doc_processing/loader.py`):**
    -   [ ] Create function `load_pdf_documents(file_paths: list[Path]) -> list[Document]` using `PyPDFLoader`. Handle potential errors.
-   [ ] **Document Chunking (`rag_app/doc_processing/chunker.py`):**
    -   [ ] Create function `chunk_documents(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]` using `RecursiveCharacterTextSplitter`.

## Phase 3: Retrieval Backend (Local)

-   [ ] **Embeddings Setup (`rag_app/retrieval/retriever.py`):**
    -   [ ] Create function `initialize_embedding_model(model_name: str)` using `HuggingFaceEmbeddings`.
-   [ ] **Vector Store Setup (ChromaDB) (`rag_app/retrieval/retriever.py`):**
    -   [ ] Create function `initialize_vector_store(embedding_function, persist_directory: str)` using `Chroma`. Configure for persistence.
    -   [ ] Create function `add_documents_to_db(vector_store, docs: list[Document])`. Ensure source metadata (filename) is stored with chunks. Handle potential batching/updates.
    -   [ ] Create function `delete_document_from_db(vector_store, filename: str)` using Chroma's metadata filtering capabilities.
-   [ ] **Indexing Logic (`app.py`):**
    -   [ ] Connect document upload: After saving a file, trigger `load_pdf_documents`, `chunk_documents`, and `add_documents_to_db`. Provide user feedback (spinner, success/error messages).
-   [ ] **Deletion Logic (`app.py`):**
    -   [ ] Implement document deletion UI (e.g., select box + button).
    -   [ ] On deletion, trigger `delete_document_from_db` *before* deleting the file from disk.
-   [ ] **Retriever Setup (`rag_app/retrieval/retriever.py`):**
    -   [ ] Create function `get_retriever(vector_store, search_type="similarity", k=4)` to get a retriever object from the vector store (e.g., `vector_store.as_retriever()`). Plan for `k` parameterization. (Hybrid search integration is a later task).

## Phase 4: LLM Integration & Core RAG Pipeline

-   [ ] **LLM Setup (`rag_app/llm/generator.py`):**
    -   [ ] Create function `initialize_llm(model_name: str, base_url: Optional[str] = None)` using `Ollama`.
-   [ ] **Prompt Engineering (`rag_app/llm/generator.py`):**
    -   [ ] Design initial prompt template (`ChatPromptTemplate` or `PromptTemplate`). Include placeholders for `context` and `question` (or structured `notes`). Instruct the LLM on its role (protocol generation), required language (German), context usage (use *only* provided context), and citation requirement (`(Source: filename.pdf)`).
-   [ ] **Core RAG Chain (`rag_app/core/pipeline.py`):**
    -   [ ] Create main function `generate_protocol(notes: dict, retriever, llm, prompt_template) -> str`.
    -   [ ] Implement logic to formulate the actual query/question to the retriever based on the input `notes` (e.g., focusing on one decision point at a time might be needed).
    -   [ ] Implement the RAG chain using LangChain Expression Language (LCEL): `context -> retriever -> format_docs -> prompt -> llm -> output_parser`.
    -   [ ] Handle combining results if processing decision points individually.

## Phase 5: UI Integration & Output Handling

-   [ ] **Input Form (`app.py`):**
    -   [ ] Implement the Streamlit form (`st.form`) for capturing General Info and Decision Points based on `st.session_state.meeting_notes`.
    -   [ ] Implement logic to add/remove decision points dynamically within the form.
    -   [ ] Ensure form data is correctly captured into `st.session_state.meeting_notes` on submission.
-   [ ] **Connecting UI to Backend (`app.py`):**
    -   [ ] On form submission ("Generate Protocol Draft"):
        -   Initialize necessary components (embeddings, vector store, retriever, llm, prompt) if not already done.
        -   Call the `generate_protocol` function from `rag_app.core.pipeline`.
        -   Store the raw output (with citations) in `st.session_state.protocol_draft`.
        -   Display a spinner during generation. Handle and display errors.
-   [ ] **Output Display (`app.py`):**
    -   [ ] Implement `display_protocol` function to render the draft, parse `(Source: ...)` citations, and create clickable download buttons for found source files.
    -   [ ] Display the output using `display_protocol` when `st.session_state.protocol_draft` is populated.
-   [ ] **Final Protocol Generation (`rag_app/core/pipeline.py`, `app.py`):**
    -   [ ] Create utility function `clean_citations(text: str) -> str` to remove `(Source: ...)` patterns using regex.
    -   [ ] After generating the draft, call `clean_citations` and store the result in `st.session_state.final_protocol`.
    -   [ ] Implement the "Download Final Protocol" button (`st.download_button`) using `st.session_state.final_protocol`.
-   [ ] **Feedback Placeholders (`app.py`):**
    -   [ ] Add basic UI elements for feedback (buttons, text area). (Logging feedback is a future task).

## Phase 6: Refinement & Utilities

-   [ ] **Configuration (`config/config.yaml`, `rag_app/utils/helpers.py`):**
    -   [ ] Refine configuration loading (decide between `.env` only or `.env` + `config.yaml`). Implement helper function if needed.
-   [ ] **Logging (`rag_app/utils/helpers.py`):**
    -   [ ] Set up basic Python logging configuration. Add simple log statements at key steps (e.g., document loading, retrieval, generation start/end).
-   [ ] **Error Handling:** Review key functions and add basic `try...except` blocks for robustness.
-   [ ] **README Update:** Update `README.md` with accurate setup and usage instructions for the prototype.
-   [ ] **Code Cleanup:** Review code for clarity, comments, and adherence to basic Python style (PEP 8).

*Note: This list focuses on the initial local prototype. Hybrid search, advanced feedback logging, testing, Dockerization, and cloud migration are subsequent phases.*