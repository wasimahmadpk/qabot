import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="QABot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


logo = Image.open("assets/logo.png")

col1, col2 = st.columns([1, 5])
with col1:
    st.image(logo, width=80)
with col2:
    st.title("QABot: Document Question Answering")



from dotenv import load_dotenv
load_dotenv()  # loads .env automatically

import os
from src.loader import load_documents_from_upload
from src.indexer import create_index
from src.query_engine import get_query_engine, query_index

api_key = os.getenv("OPENAI_API_KEY")

st.markdown("### 📄 Upload Your Document")
uploaded_files = st.file_uploader("Upload documents (txt, pdf, docx)", accept_multiple_files=True)

if uploaded_files:
    docs = load_documents_from_upload(uploaded_files)
    if docs:
        # Create index for uploaded docs
        index = create_index(docs)
        query_engine = get_query_engine(index)

        st.success(f"Uploaded and indexed {len(docs)} documents!")
        st.markdown("### 🤖 Ask a Question")
        query = st.text_input("Ask a question about your uploaded documents:")
        # if query:
            
        #     response = query_index(query_engine, query)
        #     st.write(response.response)  # or response.text / response.content depending on type

        if query:
            with st.spinner("Thinking..."):
                response = query_index(query_engine, query)
            st.markdown("#### 📌 Answer:")
            st.success(response)

    else:
        st.warning("No valid documents uploaded.")
else:
    st.info("Please upload documents to get started.")
