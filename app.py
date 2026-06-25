import html
import json
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from PIL import Image

from src.evaluation import (
    load_qa_pairs,
    parse_qa_pairs_json,
    run_evaluation,
    run_ragas_evaluation,
)
from src.indexer import create_index
from src.loader import load_documents_from_upload
from src.query_engine import get_query_engine, query_index
from src.rag_config import (
    CHUNK_STRATEGIES,
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
    initial_sidebar_state="collapsed",
)


def inject_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&display=swap');

        html, body, [class*="css"] {
            font-family: 'DM Sans', system-ui, sans-serif;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 820px;
        }

        header[data-testid="stHeader"] {
            background: transparent;
        }

        .app-header {
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 2rem;
        }

        .app-title {
            font-size: 1.65rem;
            font-weight: 700;
            letter-spacing: -0.03em;
            color: #0f172a;
            margin: 0;
            line-height: 1.2;
        }

        .app-tagline {
            color: #64748b;
            font-size: 0.92rem;
            margin: 0.2rem 0 0 0;
        }

        .empty-state {
            text-align: center;
            padding: 3rem 1.5rem;
            border: 1px dashed #cbd5e1;
            border-radius: 20px;
            background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%);
            color: #64748b;
        }

        .empty-state strong {
            display: block;
            color: #334155;
            font-size: 1.05rem;
            margin-bottom: 0.35rem;
        }

        .answer-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 1.35rem 1.5rem;
            line-height: 1.65;
            color: #1e293b;
            font-size: 1rem;
            box-shadow: 0 4px 24px rgba(15, 23, 42, 0.06);
        }

        .answer-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 1rem;
        }

        .meta-pill {
            display: inline-block;
            background: #f1f5f9;
            color: #475569;
            font-size: 0.78rem;
            font-weight: 500;
            padding: 0.25rem 0.65rem;
            border-radius: 999px;
        }

        .source-card {
            background: #fafbfc;
            border: 1px solid #e8edf2;
            border-radius: 12px;
            padding: 0.85rem 1rem;
            margin-bottom: 0.65rem;
        }

        .source-meta {
            color: #64748b;
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.01em;
            margin-bottom: 0.4rem;
            text-transform: uppercase;
        }

        .source-text {
            color: #334155;
            font-size: 0.88rem;
            line-height: 1.5;
            white-space: pre-wrap;
        }

        .status-dot {
            display: inline-block;
            width: 7px;
            height: 7px;
            border-radius: 50%;
            margin-right: 6px;
            vertical-align: middle;
        }

        .status-ok { background: #22c55e; }
        .status-warn { background: #f59e0b; }
        .status-bad { background: #ef4444; }

        div[data-testid="stMetric"] {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 0.65rem 0.85rem;
        }

        div[data-testid="stMetric"] label {
            font-size: 0.75rem !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            background: #f1f5f9;
            padding: 4px;
            border-radius: 12px;
            width: fit-content;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 9px;
            padding: 0.45rem 1.1rem;
            font-weight: 500;
        }

        .stTabs [aria-selected="true"] {
            background: #ffffff !important;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08);
        }

        section[data-testid="stSidebar"] {
            background: #fafbfc;
        }

        div[data-testid="stFileUploader"] section {
            border-radius: 14px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_session_defaults():
    for key, value in default_rag_settings().items():
        st.session_state.setdefault(key, value)
    st.session_state.setdefault("eval_set_name", "eval/qa_pairs.json (default)")
    st.session_state.setdefault("example_picker", "— pick a sample question —")


def current_rag_settings():
    return {
        "chunk_size": int(st.session_state.chunk_size),
        "chunk_overlap": int(st.session_state.chunk_overlap),
        "chunk_strategy": st.session_state.chunk_strategy,
        "top_k": int(st.session_state.top_k),
    }


def render_header():
    logo = Image.open("assets/logo.png")
    col_logo, col_text = st.columns([1, 7], vertical_alignment="center")
    with col_logo:
        st.image(logo, width=56)
    with col_text:
        st.markdown(
            """
            <div>
                <p class="app-title">QABot</p>
                <p class="app-tagline">Grounded answers from your documents</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_engineer_settings():
    with st.sidebar.expander("Advanced settings", expanded=False):
        st.selectbox(
            "Chunk strategy",
            options=list(CHUNK_STRATEGIES.keys()),
            format_func=lambda key: key,
            key="chunk_strategy",
        )
        c1, c2 = st.columns(2)
        with c1:
            st.number_input("Size", min_value=128, max_value=4096, step=64, key="chunk_size")
        with c2:
            st.number_input("Overlap", min_value=0, max_value=1024, step=16, key="chunk_overlap")
        st.number_input(
            "Top-k",
            min_value=1,
            max_value=15,
            step=1,
            key="top_k",
            help="Chunks retrieved per question. No re-index needed.",
        )
        if st.button("Reset defaults", use_container_width=True):
            for key, value in default_rag_settings().items():
                st.session_state[key] = value


def render_sidebar():
    api_ok = bool(os.getenv("OPENAI_API_KEY"))
    indexed = st.session_state.get("query_engine") is not None
    chunk_stats = st.session_state.get("chunk_stats", {})

    with st.sidebar:
        st.markdown("##### Status")
        dot = "status-ok" if api_ok else "status-bad"
        st.markdown(
            f'<span class="status-dot {dot}"></span>'
            f'{"API connected" if api_ok else "Missing API key"}',
            unsafe_allow_html=True,
        )
        if indexed:
            docs = st.session_state.get("doc_count", 0)
            chunks = chunk_stats.get("total_chunks", 0)
            st.markdown(
                f'<span class="status-dot status-ok"></span>{docs} docs · {chunks} chunks',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="status-dot status-warn"></span>No index yet',
                unsafe_allow_html=True,
            )

        st.divider()
        render_engineer_settings()


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
        "ragas_report",
    ):
        st.session_state.pop(key, None)


def apply_example_question():
    picked = st.session_state.example_picker
    if picked != "— pick a sample question —":
        st.session_state.question_box = picked
        st.session_state.example_picker = "— pick a sample question —"


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

    with st.spinner("Indexing…"):
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
    st.session_state.pop("ragas_report", None)
    refresh_query_engine()


def get_active_qa_pairs():
    custom_pairs = st.session_state.get("eval_qa_pairs")
    if custom_pairs is not None:
        return custom_pairs
    return load_qa_pairs()


def render_sources(sources):
    if not sources:
        return

    with st.expander(f"Sources ({len(sources)})", expanded=False):
        for index, source in enumerate(sources, start=1):
            score_text = (
                f" · {source['score']:.2f}"
                if source.get("score") is not None
                else ""
            )
            st.markdown(
                f"""
                <div class="source-card">
                    <div class="source-meta">
                        {source['file_name']} · chunk {source['chunk_id']}{score_text}
                    </div>
                    <div class="source-text">{html.escape(source['text'])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_qa_tab():
    uploaded_files = st.file_uploader(
        "Documents",
        accept_multiple_files=True,
        type=["pdf", "txt", "docx"],
        label_visibility="collapsed",
        help="PDF, TXT, or DOCX",
    )

    if not uploaded_files:
        clear_index_state()
        st.markdown(
            """
            <div class="empty-state">
                <strong>Upload documents to begin</strong>
                Drop PDF, TXT, or DOCX files above to build your knowledge base.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    index_uploads(uploaded_files)
    query_engine = st.session_state.get("query_engine")

    if not query_engine:
        st.warning("Could not parse any documents from your upload.")
        return

    if "question_box" not in st.session_state:
        st.session_state.question_box = ""

    st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)

    st.selectbox(
        "Examples",
        options=["— pick a sample question —", *SAMPLE_QUESTIONS],
        key="example_picker",
        on_change=apply_example_question,
        label_visibility="collapsed",
    )

    query = st.text_area(
        "Question",
        height=88,
        placeholder="Ask anything about your documents…",
        key="question_box",
        label_visibility="collapsed",
    )

    if st.button("Ask", type="primary", use_container_width=False):
        if query.strip():
            with st.spinner("Thinking…"):
                result = query_index(query_engine, query.strip())
            st.session_state.last_result = result

    result = st.session_state.get("last_result")
    if not result:
        return

    st.markdown("<div style='height: 1.25rem'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="answer-card">
            {html.escape(result['answer'])}
            <div class="answer-meta">
                <span class="meta-pill">{result['latency_ms']:.0f} ms</span>
                <span class="meta-pill">{len(result['sources'])} sources</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_sources(result["sources"])


def render_eval_tab():
    query_engine = st.session_state.get("query_engine")
    if not query_engine:
        st.markdown(
            """
            <div class="empty-state">
                <strong>Index documents first</strong>
                Upload files on the Ask tab, then return here to run evaluation.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    active_pairs = get_active_qa_pairs()
    eval_name = st.session_state.get("eval_set_name", "default")

    run_col, ragas_col, info_col = st.columns([1, 1, 2], vertical_alignment="center")
    with run_col:
        if st.button("Run evaluation", type="primary", use_container_width=True):
            with st.spinner("Evaluating…"):
                report = run_evaluation(
                    query_engine,
                    query_index,
                    qa_pairs=active_pairs,
                )
            st.session_state.eval_report = report
    with ragas_col:
        if st.button("Run RAGAS", use_container_width=True):
            with st.spinner("Running RAGAS (LLM judge)…"):
                try:
                    st.session_state.ragas_report = run_ragas_evaluation(
                        query_engine,
                        query_index,
                        qa_pairs=active_pairs,
                    )
                except ImportError as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.error(f"RAGAS evaluation failed: {exc}")
    with info_col:
        st.caption(
            f"{len(active_pairs)} questions · {eval_name} · RAGAS uses extra OpenAI calls"
        )

    with st.expander("Eval set & downloads", expanded=False):
        dl1, dl2 = st.columns(2)
        with dl1:
            sample_path = Path("eval/sample_policy.txt")
            if sample_path.exists():
                st.download_button(
                    "Sample handbook",
                    data=sample_path.read_bytes(),
                    file_name="sample_policy.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
        with dl2:
            if DEFAULT_EVAL_PATH.exists():
                st.download_button(
                    "Default eval JSON",
                    data=DEFAULT_EVAL_PATH.read_bytes(),
                    file_name="qa_pairs.json",
                    mime="application/json",
                    use_container_width=True,
                )

        uploaded_eval = st.file_uploader("Upload eval JSON", type=["json"], key="eval_json_upload")
        if uploaded_eval is not None:
            try:
                qa_pairs = parse_qa_pairs_json(uploaded_eval.getvalue().decode("utf-8"))
                st.session_state.eval_qa_pairs = qa_pairs
                st.session_state.eval_set_name = uploaded_eval.name
                st.session_state.pop("eval_report", None)
                st.session_state.pop("ragas_report", None)
                st.success(f"Loaded {len(qa_pairs)} questions.")
            except (ValueError, json.JSONDecodeError) as exc:
                st.error(str(exc))

        eval_json_text = st.text_area(
            "Edit JSON",
            value=json.dumps(get_active_qa_pairs(), indent=2),
            height=180,
            key="eval_json_editor",
        )
        btn1, btn2 = st.columns(2)
        with btn1:
            if st.button("Apply", use_container_width=True):
                try:
                    qa_pairs = parse_qa_pairs_json(eval_json_text)
                    st.session_state.eval_qa_pairs = qa_pairs
                    st.session_state.eval_set_name = "Custom eval JSON"
                    st.session_state.pop("eval_report", None)
                    st.session_state.pop("ragas_report", None)
                    st.success(f"Applied {len(qa_pairs)} questions.")
                except (ValueError, json.JSONDecodeError) as exc:
                    st.error(str(exc))
        with btn2:
            if st.button("Reset default", use_container_width=True):
                st.session_state.pop("eval_qa_pairs", None)
                st.session_state.eval_set_name = "eval/qa_pairs.json (default)"
                st.session_state.pop("eval_report", None)
                st.session_state.pop("ragas_report", None)
                st.success("Restored default set.")

    report = st.session_state.get("eval_report")
    ragas_report = st.session_state.get("ragas_report")

    if not report and not ragas_report:
        return

    if report:
        st.markdown("##### Keyword evaluation")
        summary = report["summary"]
        cols = st.columns(4)
        cols[0].metric("Questions", summary["total"])
        cols[1].metric("Latency", f"{summary['avg_latency_ms']} ms")
        cols[2].metric("Retrieval", f"{summary['retrieval_hit_rate']:.0%}")
        cols[3].metric("Grounded", f"{summary['answer_grounded_rate']:.0%}")

        with st.expander("Keyword results", expanded=False):
            display_rows = []
            for row in report["results"]:
                display_rows.append(
                    {
                        "Question": row["question"],
                        "ms": row["latency_ms"],
                        "Retrieval": "✓" if row["retrieval_hit"] else "✗",
                        "Grounded": "✓" if row["answer_grounded"] else "✗",
                        "Preview": row["answer_preview"],
                    }
                )
            st.dataframe(display_rows, use_container_width=True, hide_index=True)

    if ragas_report:
        st.markdown("##### RAGAS (LLM-as-judge)")
        ragas_summary = ragas_report["summary"]
        metric_keys = ragas_report.get("metrics", [])
        metric_cols = st.columns(len(metric_keys) + 1)
        metric_cols[0].metric("Latency", f"{ragas_summary['avg_latency_ms']} ms")
        for index, key in enumerate(metric_keys, start=1):
            label = key.replace("_", " ").title()
            metric_cols[index].metric(label, f"{ragas_summary[key]:.0%}")

        with st.expander("RAGAS results", expanded=False):
            ragas_rows = []
            for row in ragas_report["results"]:
                entry = {
                    "Question": row["question"],
                    "ms": row["latency_ms"],
                    "Preview": row["answer_preview"],
                }
                for key in metric_keys:
                    entry[key.replace("_", " ").title()] = (
                        f"{row[key]:.0%}" if key in row else "—"
                    )
                ragas_rows.append(entry)
            st.dataframe(ragas_rows, use_container_width=True, hide_index=True)


init_session_defaults()
inject_styles()
render_header()

if not os.getenv("OPENAI_API_KEY"):
    st.error("Add `OPENAI_API_KEY` to your `.env` file.")

render_sidebar()

tab_qa, tab_eval = st.tabs(["Ask", "Evaluate"])

with tab_qa:
    render_qa_tab()

with tab_eval:
    render_eval_tab()
