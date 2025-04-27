import streamlit as st
from dotenv import load_dotenv
import os
from pathlib import Path
from rag_app.utils.helpers import list_uploaded_files # Import the helper function

# --- Page Configuration ---
st.set_page_config(
    page_title="its_RAG_realestate - Protocol Assistant",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📝 its_RAG_realestate - Protocol Assistant")
st.caption("Generate real estate meeting protocols using RAG.")

# --- Load Environment Variables ---
load_dotenv()
# Example: Accessing an environment variable
# ollama_model = os.getenv("OLLAMA_MODEL")
# st.write(f"Using Ollama model: {ollama_model}") # Optional: for debugging

# --- Initialize Session State ---
# Initialize meeting notes structure if not already in state
if 'meeting_notes' not in st.session_state:
    st.session_state.meeting_notes = {
        "general_info": {
            "topic": "",
            "date": "",
            "attendees": ""
        },
        "decision_points": [] # List of dicts, e.g., {"point": "", "decision": ""}
    }

# Initialize protocol draft if not already in state
if 'protocol_draft' not in st.session_state:
    st.session_state.protocol_draft = ""

# --- Sidebar ---
with st.sidebar:
    st.header("📄 Document Management")
    st.write("Upload PDF documents to build the knowledge base.")

    # Get documents directory from environment variables
    documents_dir_str = os.getenv("DOCUMENTS_DIR", "documents/")
    documents_dir = Path(documents_dir_str)
    documents_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists

    uploaded_file = st.file_uploader(
        "Upload a PDF document", type="pdf", accept_multiple_files=False
    )

    if uploaded_file is not None:
        # Construct the full path to save the file
        file_path = documents_dir / uploaded_file.name
        try:
            # Write the uploaded file's content to the target file path
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Successfully saved '{uploaded_file.name}' to {documents_dir_str}")
            # Clear the uploader after successful save to prevent resubmission on rerun
            # Note: Streamlit's file_uploader doesn't have a direct clear method.
            # Rerunning the script usually handles this, but managing state might be needed for complex flows.
        except Exception as e:
            st.error(f"Error saving file: {e}")

    st.divider()
    st.header("📚 Knowledge Base")

    # List uploaded files
    pdf_files = list_uploaded_files(documents_dir)

    if not pdf_files:
        st.info("No PDF documents found in the knowledge base yet.")
    else:
        st.write(f"Found {len(pdf_files)} document(s):")
        for pdf_file in pdf_files:
            col1, col2 = st.columns([0.7, 0.3]) # Adjust column ratio as needed
            with col1:
                st.write(f"📄 {pdf_file.name}")
            with col2:
                try:
                    with open(pdf_file, "rb") as fp:
                        st.download_button(
                            label="Download",
                            data=fp,
                            file_name=pdf_file.name,
                            mime="application/pdf",
                            key=f"download_{pdf_file.name}" # Unique key per button
                        )
                except Exception as e:
                    st.error(f"Error reading {pdf_file.name} for download: {e}")
            # TODO: Add deletion button logic here later

# --- Main Area ---
col1, col2 = st.columns(2)

with col1:
    st.header("📋 Input Notes")
    st.write("Enter meeting details and decision points here.")
    # TODO: Add input form

with col2:
    st.header("📜 Protocol Draft")
    st.write("The generated protocol draft will appear here.")
    # TODO: Add output display area

# --- Placeholder for future logic ---
# st.write("App structure initialized. More features coming soon!") # Removed placeholder