import streamlit as st
from dotenv import load_dotenv
import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

from rag_app.utils.helpers import list_uploaded_files
from rag_app.doc_processing.loader import load_pdf_documents
from rag_app.doc_processing.chunker import chunk_documents
from rag_app.retrieval.retriever import (
    initialize_embedding_model,
    initialize_vector_store,
    add_documents_to_db,
    delete_document_from_db,
    get_retriever # Import retriever function
)
from rag_app.llm.generator import initialize_llm, rag_prompt # Import LLM init and prompt
from rag_app.core.pipeline import generate_protocol, clean_citations # Import core pipeline function AND cleaning function

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
            "title": "",
            "date": None,
            "property_address": "",
            "meeting_place": "",
            "start_time": None,  # Use None for time inputs
            "end_time": None,    # Use None for time inputs
            "chairperson": "",
            "secretary": "",
        },
        "decision_points": [] # List of dicts, e.g., {"point": "", "decision": ""}
    }

# Initialize protocol draft if not already in state
if 'protocol_draft' not in st.session_state:
    st.session_state.protocol_draft = ""
# Initialize final protocol if not already in state
if 'final_protocol' not in st.session_state:
    st.session_state.final_protocol = ""

# --- Caching Initialization Functions ---
# Cache embedding model initialization
@st.cache_resource
def get_embedding_model():
    st.write("Initializing embedding model...") # Debug message
    model = initialize_embedding_model()
    st.write("Embedding model initialized.") # Debug message
    return model

# Cache vector store initialization
@st.cache_resource
def get_vector_store(_embedding_model): # Pass model to ensure dependency
    st.write("Initializing vector store...") # Debug message
    vector_store = initialize_vector_store(embedding_function=_embedding_model)
    st.write("Vector store initialized.") # Debug message
    return vector_store

with st.sidebar:
    st.header("📄 Document Management")
    st.write("Upload PDF documents to build the knowledge base.")

    # Track processed files in session state
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []

    # Get documents directory from environment variables
    documents_dir_str = os.getenv("DOCUMENTS_DIR", "documents/")
    documents_dir = Path(documents_dir_str)
    documents_dir.mkdir(parents=True, exist_ok=True)

    uploaded_file = st.file_uploader(
        "Upload a PDF document", type="pdf", accept_multiple_files=False
    )

    if uploaded_file is not None:
        filename = uploaded_file.name
        # Only process new files
        if filename not in st.session_state.processed_files:
            file_path = documents_dir / filename
            try:
                # Save upload to disk
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"Successfully saved '{filename}' to {documents_dir_str}")

                # --- Indexing Logic ---
                try:
                    with st.spinner(f"Processing and indexing '{filename}'..."):
                        # 1. Initialize models and vector store (cached)
                        embedding_model = get_embedding_model()
                        vector_store = get_vector_store(embedding_model)

                        # 2. Load the newly uploaded document
                        loaded_docs = load_pdf_documents([file_path])
                        if not loaded_docs:
                            st.error(f"Could not load document content from {filename}.")
                            raise Exception("Document loading failed.")

                        # 3. Chunk the document
                        chunk_size = int(os.getenv("CHUNK_SIZE", 1000))
                        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 200))
                        chunked_docs = chunk_documents(
                            loaded_docs,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                        )
                        if not chunked_docs:
                            st.error(f"Could not chunk document {filename}.")
                            raise Exception("Document chunking failed.")

                        # 4. Add chunks to the vector store
                        add_documents_to_db(vector_store, chunked_docs)

                    st.success(f"Successfully indexed '{filename}'!")
                    # Mark this file as processed
                    st.session_state.processed_files.append(filename)

                except Exception as index_e:
                    st.error(f"Failed to process or index '{filename}': {index_e}")

            except Exception as e:
                st.error(f"Error saving file: {e}")
        else:
            st.info(f"'{filename}' has already been processed this session.")

    st.divider()
    st.header("📚 Knowledge Base")

    # List uploaded files
    pdf_files = list_uploaded_files(documents_dir)

    if not pdf_files:
        st.info("No PDF documents found in the knowledge base yet.")
    else:
        st.write(f"Found {len(pdf_files)} document(s):")
        for pdf_file in pdf_files:
            col1, col2 = st.columns([0.7, 0.3])
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
                            key=f"download_{pdf_file.name}"
                        )
                except Exception as e:
                    st.error(f"Error reading {pdf_file.name} for download: {e}")

        st.divider()
        st.subheader("🗑️ Delete Document")
        if pdf_files:
            file_to_delete = st.selectbox(
                "Select document to delete:",
                options=[f.name for f in pdf_files],
                placeholder="Choose a file..."
            )

            if st.button("Delete Selected Document", type="primary", disabled=(file_to_delete is None)):
                if file_to_delete:
                    file_path_to_delete = documents_dir / file_to_delete
                    try:
                        with st.spinner(f"Deleting '{file_to_delete}' and its index..."):
                            # 1. Initialize models and vector store (cached)
                            embedding_model = get_embedding_model()
                            vector_store = get_vector_store(embedding_model)

                            # 2. Delete from vector store first
                            delete_document_from_db(vector_store, file_to_delete)

                            # 3. Delete the file from disk
                            file_path_to_delete.unlink()

                        st.success(f"Successfully deleted '{file_to_delete}' and its index.")
                        st.rerun()

                    except FileNotFoundError:
                        st.error(f"Error: File '{file_to_delete}' not found on disk.")
                        try:
                            delete_document_from_db(vector_store, file_to_delete)
                            st.info(f"Removed index entries for '{file_to_delete}'.")
                        except Exception as db_del_e:
                            st.error(f"Also failed to remove index entries: {db_del_e}")
                        st.rerun()
                    except Exception as del_e:
                        st.error(f"Failed to delete '{file_to_delete}': {del_e}")


# --- Main Area ---
col1, col2 = st.columns(2)

with col1:
    st.header("📋 Eingabe Notizen")
    st.write("Angaben zur Versammlung und den besprochenen Punkten.")

    # Use a form to gather all inputs before processing
    with st.form("protocol_input_form",enter_to_submit=False):
        st.subheader("Allgemeine Informationen")
        title = st.text_input("Titel:", value=st.session_state.meeting_notes["general_info"].get("title", ""))
        property_address = st.text_input("Objektadresse:", value=st.session_state.meeting_notes["general_info"].get("property_address", ""))
        date = st.date_input("Datum:", value=st.session_state.meeting_notes["general_info"].get("date", ""))
        meeting_place = st.text_input("Besprechungsort:", value=st.session_state.meeting_notes["general_info"].get("meeting_place", ""))
        start_time = st.time_input("Beginn:", value=st.session_state.meeting_notes["general_info"].get("start_time", ""))
        end_time = st.time_input("Ende:", value=st.session_state.meeting_notes["general_info"].get("end_time", ""))
        chairperson = st.text_input("Versammlungsleiter:", value=st.session_state.meeting_notes["general_info"].get("chairperson", ""))
        secretary = st.text_input("Protokollführer:", value=st.session_state.meeting_notes["general_info"].get("secretary", ""))
    

        st.divider()
        st.subheader("Besprochene Punkte & Entscheidungen")

        # Display existing decision points
        decision_points = st.session_state.meeting_notes["decision_points"]
        for i, point_data in enumerate(decision_points):
            st.markdown(f"**Punkt {i+1}**")
            point = st.text_area(f"Beschreibung Punkt {i+1}", value=point_data.get("point", ""), key=f"point_{i}", height=100)
            decision = st.text_area(f"Entscheidung/Ergebnis Punkt {i+1}", value=point_data.get("decision", ""), key=f"decision_{i}", height=100)
            st.markdown("---") # Separator between points

        # Buttons to add/remove points - placed inside the form
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            add_point = st.form_submit_button("➕ Weiteren Punkt hinzufügen")
        with col_btn2:
            remove_point = st.form_submit_button("➖ Letzten Punkt entfernen", disabled=not decision_points) # Disable if no points

        st.divider()

        # Main submit button for the form
        submitted = st.form_submit_button("🚀 Protokoll generieren")

        # --- Form Submission Logic ---
        if submitted or add_point or remove_point:
            # Always update general info from the form fields
            st.session_state.meeting_notes["general_info"]["topic"] = title
            st.session_state.meeting_notes["general_info"]["date"] = date
            st.session_state.meeting_notes["general_info"]["property_address"] = property_address
            st.session_state.meeting_notes["general_info"]["meeting_place"] = meeting_place
            st.session_state.meeting_notes["general_info"]["start_time"] = start_time
            st.session_state.meeting_notes["general_info"]["end_time"] = end_time
            st.session_state.meeting_notes["general_info"]["chairperson"] = chairperson
            st.session_state.meeting_notes["general_info"]["secretary"] = secretary

            

            # Update existing decision points from form fields
            # Need to access form elements by key - st.session_state holds widget values directly
            updated_points = []
            for i in range(len(decision_points)):
                 updated_points.append({
                     "point": st.session_state[f"point_{i}"],
                     "decision": st.session_state[f"decision_{i}"]
                 })
            st.session_state.meeting_notes["decision_points"] = updated_points

            # Handle Add/Remove Point Actions
            if add_point:
                st.session_state.meeting_notes["decision_points"].append({"point": "", "decision": ""})
                # Rerun immediately to show the new empty fields within the form
                st.rerun()
            elif remove_point:
                if st.session_state.meeting_notes["decision_points"]:
                    st.session_state.meeting_notes["decision_points"].pop()
                    # Rerun immediately to remove the last fields from the form
                    st.rerun()
            elif submitted:
                # Ensure there are notes to process
                if not st.session_state.meeting_notes["general_info"]["topic"] and not st.session_state.meeting_notes["decision_points"]:
                    st.warning("Please enter some meeting notes before generating the protocol.")
                else:
                    # --- Trigger RAG Pipeline ---
                    try:
                        with st.spinner("🧠 Generating protocol draft... This may take a moment."):
                            # 1. Initialize components (some cached)
                            embedding_model = get_embedding_model()
                            vector_store = get_vector_store(embedding_model)
                            retriever = get_retriever(vector_store) # Create retriever
                            llm = initialize_llm() # Initialize LLM

                            # 2. Call the generation function
                            generated_draft = generate_protocol(
                                notes=st.session_state.meeting_notes,
                                retriever=retriever,
                                llm=llm,
                                prompt_template=rag_prompt
                            )

                            # 3. Store raw draft result in session state
                            st.session_state.protocol_draft = generated_draft

                            # 4. Clean citations and store final protocol
                            st.session_state.final_protocol = clean_citations(generated_draft)

                            st.success("Protocol draft generated successfully!")
                            # No rerun needed here, output will display in col2 based on session state change

                    except Exception as gen_e:
                        st.error(f"An error occurred during protocol generation: {gen_e}")
                        st.session_state.protocol_draft = "" # Clear any previous draft on error
                        st.session_state.final_protocol = "" # Also clear final protocol on error

# --- Output Column ---
with col2:
    st.header("📜 Protocol Draft")
    # Display the generated draft if available in session state
    if st.session_state.protocol_draft:
        st.markdown("---") # Add a separator
        st.subheader("Generated Draft (with Citations):")
        # Use markdown to allow for potential formatting from the LLM
        # Use a text area for better scrollability and copy-paste
        st.text_area("Draft Output", value=st.session_state.protocol_draft, height=400, disabled=True)

        # Add download button for the final, cleaned protocol
        if st.session_state.final_protocol:
            st.download_button(
                label="⬇️ Download Final Protocol (Cleaned)",
                data=st.session_state.final_protocol.encode('utf-8'), # Encode to bytes
                file_name="final_protocol.txt", # Suggest a filename
                mime="text/plain",
                key="download_final"
            )
        st.divider()
        st.subheader("Feedback (Optional)")
        feedback_text = st.text_area("Provide feedback on the generated draft:", key="feedback_text")
        if st.button("Submit Feedback", key="submit_feedback", disabled=not feedback_text):
            # TODO: Implement feedback logging/storage logic here in the future
            st.toast("Feedback submitted (Placeholder - not saved yet).")

    else:
        st.info("The generated protocol draft will appear here once you submit the notes.")

# --- Placeholder for future logic ---
# st.write("App structure initialized. More features coming soon!") # Removed placeholder