import html
import json
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from PIL import Image

from src.evaluation import load_qa_pairs, parse_qa_pairs_json, run_evaluation
from src.indexer import create_index
from src.loader import load_documents_from_upload
from src.query_engine import get_query_engine, query_index
from src.rag_config import (
    CHUNK_STRATEGIES,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_STRATEGY,
    default_rag_settings,
    index_config_key,
)
from src.upload_cache import upload_signature

load_dotenv()

SAMPLE_QUESTIONS = [
    "How many remote days per week are allowed?",
    "What are the core hours on remote days?",
    "How should employees access systems remotely?",
    "What is the minimum password length?",
    "What is the capital of France?",
]

DEFAULT_EVAL_PATH = Path("eval/qa_pairs.json")


st.set_page_config(
    page_title="QABot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles():
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.5rem;
            max-width: 1100px;
        }
        .hero {
            background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
            color: #f8fafc;
            padding: 1.25rem 1.5rem;
            border-radius: 16px;
            margin-bottom: 1rem;
        }
        .hero h1 {
            color: #f8fafc;
            font-size: 1.75rem;
            margin: 0 0 0.35rem 0;
        }
        .hero p {
            color: #cbd5e1;
            margin: 0;
            font-size: 0.95rem;
        }
        .stat-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 0.9rem 1rem;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
        }
        .source-card {
            background: #f8fafc;
            border-left: 4px solid #2563eb;
            border-radius: 8px;
            padding: 0.75rem 1rem;
            margin-bottom: 0.75rem;
        }
        .source-meta {
            color: #475569;
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 0.35rem;
        }
        .source-text {
            color: #334155;
            font-size: 0.92rem;
            line-height: 1.45;
            white-space: pre-wrap;
        }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 0.75rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0 0;
            padding: 0.5rem 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_session_defaults():
    for key, value in default_rag_settings().items():
        st.session_state.setdefault(key, value)
    st.session_state.setdefault("eval_set_name", "eval/qa_pairs.json (default)")


def current_rag_settings():
    return {
        "chunk_size": int(st.session_state.chunk_size),
        "chunk_overlap": int(st.session_state.chunk_overlap),
        "chunk_strategy": st.session_state.chunk_strategy,
        "top_k": int(st.session_state.top_k),
    }


def render_header():
    logo = Image.open("assets/logo.png")
    left, right = st.columns([1, 8])
    with left:
        st.image(logo, width=72)
    with right:
        st.markdown(
            """
            <div class="hero">
                <h1>QABot — Enterprise RAG Assistant</h1>
                <p>Upload documentation, retrieve grounded answers with citations,
                and measure RAG quality — chunk → ChromaDB → LLM.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_engineer_settings():
    with st.sidebar.expander("RAG settings (engineer)", expanded=True):
        st.caption("Chunk changes trigger re-indexing on the next upload refresh.")

        st.selectbox(
            "Chunk strategy",
            options=list(CHUNK_STRATEGIES.keys()),
            format_func=lambda key: f"{key} — {CHUNK_STRATEGIES[key]}",
            key="chunk_strategy",
        )
        st.number_input(
            "Chunk size",
            min_value=128,
            max_value=4096,
            step=64,
            key="chunk_size",
        )
        st.number_input(
            "Chunk overlap",
            min_value=0,
            max_value=1024,
            step=16,
            key="chunk_overlap",
        )
        st.number_input(
            "Top-k retrieval",
            min_value=1,
            max_value=15,
            step=1,
            key="top_k",
            help="Number of chunks retrieved per question. Updates immediately without re-indexing.",
        )

        if st.button("Reset RAG defaults", use_container_width=True):
            defaults = default_rag_settings()
            for key, value in defaults.items():
                st.session_state[key] = value


def render_sidebar():
    api_ok = bool(os.getenv("OPENAI_API_KEY"))
    indexed = st.session_state.get("query_engine") is not None
    chunk_stats = st.session_state.get("chunk_stats", {})
    settings = current_rag_settings()

    with st.sidebar:
        render_engineer_settings()

        st.divider()
        st.markdown("### System status")
        st.markdown("✅ OpenAI connected" if api_ok else "❌ Missing `OPENAI_API_KEY`")
        st.markdown("✅ Index ready" if indexed else "⏳ Waiting for documents")

        if indexed:
            st.metric("Documents", st.session_state.get("doc_count", 0))
            st.metric("Chunks indexed", chunk_stats.get("total_chunks", 0))

        st.divider()
        st.markdown("### Active RAG config")
        st.markdown(
            f"""
            - Strategy: **{settings['chunk_strategy']}**  
            - Size / overlap: **{settings['chunk_size']} / {settings['chunk_overlap']}**  
            - Top-k: **{settings['top_k']}**
            """
        )


def clear_index_state():
    for key in (
        "upload_sig",
        "index_config_key",
        "vector_index",
        "query_engine",
        "active_top_k",
        "doc_count",
        "chunk_stats",
        "last_result",
        "eval_report",
    ):
        st.session_state.pop(key, None)


def set_sample_question(question):
    st.session_state.question_box = question


def refresh_query_engine():
    index = st.session_state.get("vector_index")
    if index is None:
        return
    settings = current_rag_settings()
    st.session_state.query_engine = get_query_engine(index, settings["top_k"])
    st.session_state.active_top_k = settings["top_k"]


def index_uploads(uploaded_files):
    settings = current_rag_settings()
    sig = upload_signature(uploaded_files)
    config_key = index_config_key(sig, settings)

    if (
        st.session_state.get("index_config_key") == config_key
        and st.session_state.get("vector_index") is not None
    ):
        if st.session_state.get("active_top_k") != settings["top_k"]:
            refresh_query_engine()
        return

    docs = load_documents_from_upload(uploaded_files)
    if not docs:
        clear_index_state()
        return

    with st.spinner("Chunking, embedding, and storing in ChromaDB..."):
        index, stats = create_index(
            docs,
            config_key,
            chunk_size=settings["chunk_size"],
            chunk_overlap=settings["chunk_overlap"],
            chunk_strategy=settings["chunk_strategy"],
        )

    st.session_state.upload_sig = sig
    st.session_state.index_config_key = config_key
    st.session_state.vector_index = index
    st.session_state.doc_count = len(docs)
    st.session_state.chunk_stats = stats
    st.session_state.pop("last_result", None)
    st.session_state.pop("eval_report", None)
    refresh_query_engine()


def get_active_qa_pairs():
    custom_pairs = st.session_state.get("eval_qa_pairs")
    if custom_pairs is not None:
        return custom_pairs
    return load_qa_pairs()


def render_index_status(chunk_stats):
    cols = st.columns(5)
    cols[0].metric("Documents", st.session_state.get("doc_count", 0))
    cols[1].metric("Total chunks", chunk_stats.get("total_chunks", 0))
    cols[2].metric("Chunk size", chunk_stats.get("chunk_size", DEFAULT_CHUNK_SIZE))
    cols[3].metric("Overlap", chunk_stats.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP))
    cols[4].metric("Strategy", chunk_stats.get("chunk_strategy", DEFAULT_CHUNK_STRATEGY))

    if chunk_stats.get("chunks_by_file"):
        with st.expander("Chunk breakdown by file", expanded=False):
            for file_name, count in chunk_stats["chunks_by_file"].items():
                st.markdown(f"- **{file_name}** — {count} chunks")


def render_sources(sources):
    if not sources:
        st.caption("No source chunks returned for this query.")
        return

    st.markdown("#### Retrieved sources")
    for index, source in enumerate(sources, start=1):
        score_text = (
            f" · score {source['score']:.3f}"
            if source.get("score") is not None
            else ""
        )
        st.markdown(
            f"""
            <div class="source-card">
                <div class="source-meta">
                    Source {index} · {source['file_name']} · chunk {source['chunk_id']}{score_text}
                </div>
                <div class="source-text">{html.escape(source['text'])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_qa_tab():
    st.markdown("### 1 · Upload documents")
    uploaded_files = st.file_uploader(
        "Drag and drop PDF, TXT, or DOCX files",
        accept_multiple_files=True,
        type=["pdf", "txt", "docx"],
        label_visibility="collapsed",
    )

    if not uploaded_files:
        clear_index_state()
        st.markdown(
            """
            <div class="stat-card">
            👋 **Get started** — upload one or more documents above to build your knowledge base.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    index_uploads(uploaded_files)
    query_engine = st.session_state.get("query_engine")
    chunk_stats = st.session_state.get("chunk_stats", {})

    if not query_engine:
        st.warning("No valid documents could be parsed from your upload.")
        return

    st.success("Documents indexed and ready for questions.")
    render_index_status(chunk_stats)

    st.divider()
    st.markdown("### 2 · Ask a question")

    if "question_box" not in st.session_state:
        st.session_state.question_box = ""

    st.caption("Try a sample question:")
    sample_cols = st.columns(len(SAMPLE_QUESTIONS))
    for col, question in zip(sample_cols, SAMPLE_QUESTIONS):
        with col:
            st.button(
                question[:28] + ("…" if len(question) > 28 else ""),
                key=f"sample_{question}",
                use_container_width=True,
                on_click=set_sample_question,
                args=(question,),
                help=question,
            )

    query = st.text_area(
        "Your question",
        height=90,
        placeholder="e.g. How many remote days per week are allowed?",
        key="question_box",
    )

    ask_col, _ = st.columns([1, 3])
    with ask_col:
        ask_clicked = st.button("Ask QABot", type="primary", use_container_width=True)

    if ask_clicked and query.strip():
        with st.spinner("Retrieving context and generating answer..."):
            result = query_index(query_engine, query.strip())
        st.session_state.last_result = result

    result = st.session_state.get("last_result")
    if not result:
        return

    st.divider()
    st.markdown("### 3 · Answer")

    metric_cols = st.columns(3)
    metric_cols[0].metric("Latency", f"{result['latency_ms']:.0f} ms")
    metric_cols[1].metric("Sources retrieved", len(result["sources"]))
    metric_cols[2].metric("Top-k", current_rag_settings()["top_k"])

    st.markdown(
        f"""
        <div class="stat-card" style="border-left: 4px solid #16a34a;">
        {html.escape(result['answer'])}
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_sources(result["sources"])


def render_eval_tab():
    st.markdown("### RAG evaluation")
    st.write(
        "Measure latency, retrieval hit rate, and answer grounding against a golden Q&A set."
    )

    eval_cols = st.columns(2)
    with eval_cols[0]:
        sample_path = Path("eval/sample_policy.txt")
        if sample_path.exists():
            st.download_button(
                label="Download sample handbook",
                data=sample_path.read_bytes(),
                file_name="sample_policy.txt",
                mime="text/plain",
            )
    with eval_cols[1]:
        if DEFAULT_EVAL_PATH.exists():
            st.download_button(
                label="Download default eval JSON",
                data=DEFAULT_EVAL_PATH.read_bytes(),
                file_name="qa_pairs.json",
                mime="application/json",
            )

    with st.expander("Manage evaluation set (engineer)", expanded=False):
        st.caption(f"Active set: **{st.session_state.get('eval_set_name', 'default')}**")

        uploaded_eval = st.file_uploader(
            "Upload eval JSON",
            type=["json"],
            key="eval_json_upload",
        )
        if uploaded_eval is not None:
            try:
                qa_pairs = parse_qa_pairs_json(uploaded_eval.getvalue().decode("utf-8"))
                st.session_state.eval_qa_pairs = qa_pairs
                st.session_state.eval_set_name = uploaded_eval.name
                st.session_state.pop("eval_report", None)
                st.success(f"Loaded {len(qa_pairs)} evaluation questions from {uploaded_eval.name}.")
            except (ValueError, json.JSONDecodeError) as exc:
                st.error(f"Invalid eval JSON: {exc}")

        eval_json_text = st.text_area(
            "Edit eval JSON",
            value=json.dumps(get_active_qa_pairs(), indent=2),
            height=220,
            key="eval_json_editor",
        )

        editor_cols = st.columns(2)
        with editor_cols[0]:
            if st.button("Apply eval JSON", use_container_width=True):
                try:
                    qa_pairs = parse_qa_pairs_json(eval_json_text)
                    st.session_state.eval_qa_pairs = qa_pairs
                    st.session_state.eval_set_name = "Custom eval JSON"
                    st.session_state.pop("eval_report", None)
                    st.success(f"Applied {len(qa_pairs)} evaluation questions.")
                except (ValueError, json.JSONDecodeError) as exc:
                    st.error(f"Invalid eval JSON: {exc}")
        with editor_cols[1]:
            if st.button("Reset to default eval set", use_container_width=True):
                st.session_state.pop("eval_qa_pairs", None)
                st.session_state.eval_set_name = "eval/qa_pairs.json (default)"
                st.session_state.pop("eval_report", None)
                st.success("Restored default evaluation set.")

    query_engine = st.session_state.get("query_engine")
    if not query_engine:
        st.warning("Upload and index documents on the **Ask Questions** tab first.")
        return

    active_pairs = get_active_qa_pairs()
    st.caption(f"Running {len(active_pairs)} golden questions from {st.session_state.get('eval_set_name', 'default')}.")

    if st.button("Run evaluation suite", type="primary"):
        with st.spinner("Running golden Q&A evaluation..."):
            report = run_evaluation(
                query_engine,
                query_index,
                qa_pairs=active_pairs,
            )
        st.session_state.eval_report = report

    report = st.session_state.get("eval_report")
    if not report:
        st.info("Click **Run evaluation suite** after documents have been indexed.")
        return

    summary = report["summary"]
    cols = st.columns(4)
    cols[0].metric("Questions", summary["total"])
    cols[1].metric("Avg latency", f"{summary['avg_latency_ms']} ms")
    cols[2].metric(
        "Retrieval hit rate",
        f"{summary['retrieval_hit_rate']:.0%}",
        help="Did retrieved chunks contain the expected keywords?",
    )
    cols[3].metric(
        "Grounded answers",
        f"{summary['answer_grounded_rate']:.0%}",
        help="Did answers include expected facts or 'I don't know' for out-of-scope questions?",
    )

    display_rows = []
    for row in report["results"]:
        display_rows.append(
            {
                "Question": row["question"],
                "Latency (ms)": row["latency_ms"],
                "Retrieval hit": "✅" if row["retrieval_hit"] else "❌",
                "Grounded": "✅" if row["answer_grounded"] else "❌",
                "Answer preview": row["answer_preview"],
            }
        )
    st.dataframe(display_rows, use_container_width=True, hide_index=True)


init_session_defaults()
inject_styles()
render_header()

if not os.getenv("OPENAI_API_KEY"):
    st.error("Add `OPENAI_API_KEY` to your `.env` file before using QABot.")

render_sidebar()

tab_qa, tab_eval = st.tabs(["💬 Ask Questions", "📊 Evaluate RAG"])

with tab_qa:
    render_qa_tab()

with tab_eval:
    render_eval_tab()
