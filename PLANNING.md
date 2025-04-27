its_rag_realestate/
├── .env                  # Environment variables (API keys - keep this out of Git!)
├── .gitignore            # Specifies intentionally untracked files that Git should ignore
├── Dockerfile            # For building a Docker image (future scalability)
├── README.md             # Project description, setup instructions, etc.
├── app.py                # Main Streamlit application script (entry point)
├── config/
│   └── config.yaml       # Application configuration (model names, paths, etc. - optional)
├── documents/            # Stores uploaded PDF knowledge base documents
├── notes/                # Stores input notes (e.g., JSON examples or saved sessions)
├── rag_app/              # Core Python package for the RAG application logic
│   ├── __init__.py       # Makes 'rag_app' a Python package
│   ├── core/
│   │   ├── __init__.py
│   │   └── pipeline.py     # Core RAG chain/pipeline logic
│   ├── doc_processing/
│   │   ├── __init__.py
│   │   └── loader.py       # Document loading (PDFs)
│   │   └── chunker.py      # Document splitting/chunking
│   ├── llm/
│   │   ├── __init__.py
│   │   └── generator.py    # LLM interaction logic (Ollama wrapper, prompt formatting)
│   ├── retrieval/
│   │   ├── __init__.py
│   │   └── retriever.py    # Logic for embedding, vector store interaction, retrieval
│   ├── ui/
│   │   ├── __init__.py
│   │   └── components.py   # Reusable Streamlit UI components (optional)
│   └── utils/
│       ├── __init__.py
│       └── helpers.py      # Utility functions (e.g., logging setup, file helpers)
├── requirements.txt      # Python dependencies
├── tests/                # Directory for automated tests (important for production)
│   ├── __init__.py
│   └── test_pipeline.py  # Example test file
└── vector_db/            # Persistent storage for local ChromaDB (can be gitignored)
    └── # (ChromaDB files will appear here)