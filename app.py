import streamlit as st
from dotenv import load_dotenv
import os

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
    st.write("Upload and manage your knowledge base documents here.")
    # TODO: Add file uploader
    # TODO: Add document list and deletion

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