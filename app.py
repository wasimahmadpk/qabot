from dotenv import load_dotenv
load_dotenv()  # loads .env automatically

import os
import streamlit as st
from src.loader import load_documents_from_upload
from src.indexer import create_index
from src.query_engine import get_query_engine, query_index

api_key = os.getenv("OPENAI_API_KEY")

st.title("QA Bot - Upload Your Documents")

uploaded_files = st.file_uploader("Upload documents (txt, pdf, docx)", accept_multiple_files=True)

if uploaded_files:
    docs = load_documents_from_upload(uploaded_files)
    if docs:
        # Create index for uploaded docs
        index = create_index(docs)
        query_engine = get_query_engine(index)

        st.success(f"Uploaded and indexed {len(docs)} documents!")

        query = st.text_input("Ask a question about your uploaded documents:")
        if query:
            response = query_index(query_engine, query)
            st.write(response)
    else:
        st.warning("No valid documents uploaded.")
else:
    st.info("Please upload documents to get started.")
