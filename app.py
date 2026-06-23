import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="QABot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)


logo = Image.open("assets/logo.png")

col1, col2 = st.columns([1, 5])
with col1:
    st.image(logo, width=80)
with col2:
    st.title("QABot: Document Question Answering")


from dotenv import load_dotenv

load_dotenv()

import os
from pathlib import Path

from src.evaluation import run_evaluation
from src.indexer import create_index
from src.loader import load_documents_from_upload
from src.query_engine import get_query_engine, query_index
from src.upload_cache import upload_signature

if not os.getenv("OPENAI_API_KEY"):
    st.error("Set OPENAI_API_KEY in your .env file before using QABot.")

tab_qa, tab_eval = st.tabs(["Ask Questions", "Evaluate RAG"])

with tab_qa:
    st.markdown("### Upload Your Document")
    uploaded_files = st.file_uploader(
        "Upload documents (txt, pdf, docx)",
        accept_multiple_files=True,
    )

    if uploaded_files:
        sig = upload_signature(uploaded_files)
        if st.session_state.get("upload_sig") != sig:
            docs = load_documents_from_upload(uploaded_files)
            if docs:
                with st.spinner("Chunking, embedding, and indexing..."):
                    index, stats = create_index(docs, sig)
                st.session_state.upload_sig = sig
                st.session_state.query_engine = get_query_engine(index)
                st.session_state.doc_count = len(docs)
                st.session_state.chunk_stats = stats
            else:
                st.session_state.pop("upload_sig", None)
                st.session_state.pop("query_engine", None)
                st.session_state.pop("doc_count", None)
                st.session_state.pop("chunk_stats", None)

        query_engine = st.session_state.get("query_engine")
        chunk_stats = st.session_state.get("chunk_stats", {})
        if query_engine:
            st.success(
                f"Indexed {st.session_state.doc_count} documents "
                f"into {chunk_stats.get('total_chunks', '?')} chunks "
                f"(512 chars, 128 overlap). Stored in ChromaDB."
            )
            if chunk_stats.get("chunks_by_file"):
                with st.expander("Chunk breakdown by file"):
                    for file_name, count in chunk_stats["chunks_by_file"].items():
                        st.write(f"- **{file_name}**: {count} chunks")

            st.markdown("### Ask a Question")
            query = st.text_input("Ask a question about your uploaded documents:")

            if query:
                with st.spinner("Thinking..."):
                    result = query_index(query_engine, query)
                st.markdown("#### Answer")
                st.success(result["answer"])
                st.caption(f"Latency: {result['latency_ms']:.0f} ms")
                if result["sources"]:
                    with st.expander("Retrieved sources"):
                        for source in result["sources"]:
                            st.markdown(
                                f"**{source['file_name']}** (chunk {source['chunk_id']})"
                            )
                            st.text(source["text"])
        else:
            st.warning("No valid documents uploaded.")
    else:
        st.session_state.pop("upload_sig", None)
        st.session_state.pop("query_engine", None)
        st.session_state.pop("doc_count", None)
        st.session_state.pop("chunk_stats", None)
        st.info("Please upload documents to get started.")

with tab_eval:
    st.markdown("### RAG Evaluation")
    st.write(
        "Run golden Q&A checks for latency, retrieval hit rate, and grounded answers. "
        "Upload `eval/sample_policy.txt` on the Ask Questions tab first for the demo set."
    )

    sample_path = Path("eval/sample_policy.txt")
    if sample_path.exists():
        st.download_button(
            label="Download sample policy (eval demo)",
            data=sample_path.read_bytes(),
            file_name="sample_policy.txt",
            mime="text/plain",
        )

    query_engine = st.session_state.get("query_engine")
    if not query_engine:
        st.warning("Upload and index documents first (try eval/sample_policy.txt).")
    elif st.button("Run evaluation"):
        with st.spinner("Running golden Q&A evaluation..."):
            report = run_evaluation(query_engine, query_index)

        summary = report["summary"]
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Avg latency", f"{summary['avg_latency_ms']} ms")
        col_b.metric("Retrieval hit rate", f"{summary['retrieval_hit_rate']:.0%}")
        col_c.metric("Grounded answers", f"{summary['answer_grounded_rate']:.0%}")

        st.dataframe(report["results"], use_container_width=True)
