# QABot

**QABot** is a Streamlit app for **retrieval-augmented generation (RAG)** over your own documents. Upload PDF, TXT, or DOCX files, ask questions in natural language, and get **grounded answers with source citations**. Vectors persist in **ChromaDB**; embeddings and answers use **OpenAI** via **LlamaIndex**.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.45-red)
![LlamaIndex](https://img.shields.io/badge/llamaindex-0.12-green)

## What it does

| Tab | Purpose |
|-----|---------|
| **Ask Questions** | Upload docs, index into ChromaDB, query with citations |
| **Evaluate RAG** | Run a golden Q&A suite and measure retrieval + grounding |

The `src/` pipeline is UI-agnostic — the same ingest, index, and query modules can back a REST API, MCP server, or help portal.

## Features

- **Multi-file upload** — index several documents in one session (`.pdf`, `.txt`, `.docx`)
- **Tunable RAG settings** — chunk strategy, size, overlap, and top-k from the sidebar
- **Sentence-aware chunking** — respects sentence boundaries by default; token-based splitting also available
- **Grounded prompts** — answer only from retrieved context; cite `[file_name, chunk_id]`; say "I don't know" when evidence is missing
- **ChromaDB persistence** — vectors survive app restarts for the same upload + config signature
- **Upload caching** — SHA-256 signature skips re-embedding when files and chunk settings are unchanged
- **RAG evaluation suite** — latency, retrieval hit rate, and answer grounding against golden Q&A
- **Custom eval sets** — upload, edit, or reset evaluation JSON without leaving the app

## Architecture

```
Upload → parse (loader) → chunk → embed → ChromaDB
                              ↓
User question → retrieve top-k chunks → grounded prompt → OpenAI LLM → answer + sources
                              ↓
Evaluate tab → golden Q&A set → latency + retrieval + grounding metrics
```

| Step | Module | Role |
|------|--------|------|
| Config | `src/rag_config.py` | Defaults for chunk size, overlap, strategy, top-k; stable index keys |
| Load | `src/loader.py` | Parse PDF (PyMuPDF), TXT, DOCX into LlamaIndex `Document` objects |
| Chunk | `src/chunking.py` | Split documents with overlap; attach `file_name` and `chunk_id` metadata |
| Index | `src/indexer.py` | Embed chunks and persist in ChromaDB (`./chroma_db/`) |
| Prompt | `src/prompts.py` | Grounded QA templates (anti-hallucination rules + citations) |
| Query | `src/query_engine.py` | Retrieve, answer, return latency and source chunks |
| Eval | `src/evaluation.py` | Golden Q&A metrics for retrieval and answer quality |
| Cache | `src/upload_cache.py` | SHA-256 upload signature so unchanged files skip rebuild |

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/wasimahmadpk/qabot.git
cd qabot

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure OpenAI

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-your-key-here
```

### 3. Run the app

```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

### 4. Ask questions

1. Upload one or more documents on the **Ask Questions** tab.
2. Wait for the indexing success message (document and chunk counts).
3. Type a question or pick a sample prompt — review the answer, latency, and retrieved sources.

### 5. Run evaluation (optional)

1. Upload `eval/sample_policy.txt` (or your own docs) on the **Ask Questions** tab.
2. Open the **Evaluate RAG** tab.
3. Click **Run evaluation suite** to score retrieval hit rate and answer grounding.

Download buttons on the eval tab provide the sample handbook and default `qa_pairs.json`.

## RAG settings (sidebar)

Engineer controls live in the sidebar under **RAG settings (engineer)**:

| Setting | Default | Notes |
|---------|---------|-------|
| Chunk strategy | `sentence` | `sentence` respects boundaries; `token` uses fixed token windows |
| Chunk size | 512 | Triggers re-indexing when changed |
| Chunk overlap | 128 | Triggers re-indexing when changed |
| Top-k | 3 | Retrieval depth; applies immediately without re-indexing |

Changing chunk settings creates a new ChromaDB collection keyed by upload signature + config. Use **Reset RAG defaults** to restore defaults.

## Evaluation

The eval suite runs golden questions from `eval/qa_pairs.json` (or a custom set) and reports:

| Metric | Meaning |
|--------|---------|
| **Retrieval hit rate** | Did retrieved chunks contain all expected keywords? |
| **Grounded answers** | Did the answer include expected facts (or "I don't know" for out-of-scope items)? |
| **Avg latency** | End-to-end query time in milliseconds |

Each eval item supports:

```json
{
  "question": "How many PTO days can carry over to the next year?",
  "expected_keywords": ["5 days"],
  "file_name": "sample_policy.txt"
}
```

- `expected_keywords` — facts the retrieval and answer should contain
- `file_name` — optional; restrict retrieval checks to a specific source file

Upload custom JSON, edit in the text area, or reset to the default set from the **Manage evaluation set (engineer)** panel.

## Design decisions

**Chunking (512 / 128)** — balances recall (enough context per chunk) with precision (smaller chunks reduce irrelevant retrieval noise). Overlap avoids cutting sentences at boundaries.

**Grounded prompts** — the LLM must answer only from retrieved context and refuse when evidence is missing. This directly targets hallucination reduction.

**ChromaDB** — persistent vector storage keyed by upload signature and chunk config. Restarting Streamlit does not require re-embedding the same files with the same settings.

**Evaluation** — keyword-based checks are lightweight and deterministic. They are a starting point for regression testing; swap in LLM-as-judge or RAGAS-style metrics for production workloads.

## Project structure

```
qabot/
├── app.py                 # Streamlit UI (Q&A + evaluation tabs)
├── assets/logo.png
├── eval/
│   ├── qa_pairs.json      # Golden Q&A for evaluation demo
│   └── sample_policy.txt  # Enterprise-scale sample handbook
├── src/
│   ├── rag_config.py      # RAG defaults and index key helpers
│   ├── loader.py
│   ├── chunking.py
│   ├── indexer.py
│   ├── prompts.py
│   ├── query_engine.py
│   ├── evaluation.py
│   └── upload_cache.py
├── tests/
│   ├── test_rag_config.py
│   ├── test_upload_cache.py
│   ├── test_chunking.py
│   └── test_evaluation.py
├── chroma_db/             # Created at runtime (gitignored)
├── requirements.txt
└── .env                   # Not committed — add OPENAI_API_KEY locally
```

## Tests

```bash
python -m unittest discover -s tests -v
```

## Requirements

- Python **3.10+**
- **OpenAI API key** (embeddings + LLM)
- Disk space for ChromaDB and Python ML dependencies

## Limitations

- **No auth** — intended for local or trusted use only
- **PDF quality** — text extraction depends on PDF structure; scanned images need OCR (not included)
- **OpenAI by default** — swap embedding/LLM settings for local models (e.g. Ollama) in a follow-up
- **Eval keywords** — simple substring checks; not a substitute for human or LLM-based grading at scale
- **Cost** — indexing and queries use OpenAI API credits

## Dev container

A `.devcontainer/` config is included for VS Code and GitHub Codespaces.

## Contributing

Issues and pull requests are welcome on [GitHub](https://github.com/wasimahmadpk/qabot).

## Author

[wasimahmadpk](https://github.com/wasimahmadpk)
