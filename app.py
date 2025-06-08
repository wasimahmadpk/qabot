from dotenv import load_dotenv
load_dotenv()  # loads .env automatically

import os

print("OPENAI_API_KEY found?", os.getenv("OPENAI_API_KEY") is not None)


import os
import streamlit as st

from src.loader import load_documents
from src.indexer import create_index
from src.query_engine import get_query_engine, query_index

api_key = os.getenv("OPENAI_API_KEY")

# Load OpenAI key from environment variable or .env
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "your-api-key")

@st.cache_resource
def load_index():
    docs = load_documents()
    return create_index(docs)

index = load_index()
query_engine = get_query_engine(index)

st.title("QA Bot over Your Docs")

query = st.text_input("Ask something about your documents:")

if query:
    response = query_index(query_engine, query)
    st.write(response)
