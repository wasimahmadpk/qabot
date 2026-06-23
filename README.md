# QABot

**QABot** is a Streamlit app for **retrieval-augmented question answering (RAG)** over your own files. Upload PDF, TXT, or DOCX documents, ask questions in natural language, and get **grounded answers with source citations**. Vectors are persisted in **ChromaDB**; answers use **OpenAI** via **LlamaIndex**.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.45-red)
![LlamaIndex](https://img.shields.io/badge/llamaindex-0.12-green)

## Features

- **Multi-file upload** — index several documents in one session
- **Supported formats** — `.pdf`, `.txt`, `.docx`
- **Configurable chunking** — 512-character chunks with 128-character overlap
- **Grounded prompts** — answer only from context; cite `[file_name, chunk_id]`
- **ChromaDB persistence** — vectors survive app restarts (same upload set)
- **RAG evaluation tab** — latency, retrieval hit rate, grounded-answer metrics
- **Session caching** — re-indexing only when uploads change (not on every Streamlit rerun)

## Architecture

```
Upload → parse (loader) → chunk (512/128) → embed → ChromaDB
                              ↓
User question → retrieve top-k chunks → grounded prompt → OpenAI LLM → answer + sources
                              ↓
Evaluate tab → golden Q&A set → latency + retrieval + grounding metrics
```

| Step | Module | Role |
|------|--------|------|
| Load | `src/loader.py` | Parse PDF (PyMuPDF), TXT, DOCX into LlamaIndex `Document` objects |
| Chunk | `src/chunking.py` | Split documents with overlap; attach `file_name` and `chunk_id` metadata |
| Index | `src/indexer.py` | Embed chunks and persist in ChromaDB (`./chroma_db/`) |
| Prompt | `src/prompts.py` | Grounded QA templates (anti-hallucination rules + citations) |
| Query | `src/query_engine.py` | Retrieve, answer, return latency and source chunks |
| Eval | `src/evaluation.py` | Golden Q&A metrics for correctness and retrieval quality |
| Cache | `src/upload_cache.py` | SHA-256 signature so unchanged uploads skip rebuild |

## Design decisions

**Chunking (512 / 128)** — balances recall (enough context per chunk) with precision (smaller chunks reduce irrelevant retrieval noise). Overlap avoids cutting sentences at boundaries.

**Grounded prompts** — the LLM must answer only from retrieved context and say “I don't know” when evidence is missing. This directly targets hallucination reduction.

**ChromaDB** — persistent vector storage keyed by upload signature. Restarting Streamlit does not require re-embedding the same files.

**Evaluation** — a small golden set in `eval/qa_pairs.json` checks retrieval hit rate (were the right keywords retrieved?) and answer grounding (does the answer contain expected facts?). Includes an out-of-scope question to verify “I don't know” behavior.

**Integration path** — the `src/` pipeline is UI-agnostic. A Java/Eclipse online-help portal or FastAPI service would call the same ingest + query modules over REST or MCP.

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

### 4. Use the app

1. Upload one or more documents (PDF, TXT, or DOCX).
2. Confirm chunk count and ChromaDB indexing in the success message.
3. Ask a question — review the answer, latency, and retrieved sources.
4. Open the **Evaluate RAG** tab; upload `eval/sample_policy.txt` first, then run evaluation.

## Project structure

```
qabot/
├── app.py                 # Streamlit UI (Q&A + evaluation tabs)
├── assets/logo.png
├── eval/
│   ├── qa_pairs.json      # Golden Q&A for evaluation demo
│   └── sample_policy.txt  # Sample document for eval
├── src/
│   ├── loader.py
│   ├── chunking.py
│   ├── indexer.py
│   ├── prompts.py
│   ├── query_engine.py
│   ├── evaluation.py
│   └── upload_cache.py
├── tests/
│   ├── test_upload_cache.py
│   ├── test_chunking.py
│   └── test_evaluation.py
├── chroma_db/             # Created at runtime (gitignored)
├── requirements.txt
└── .env
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

- **No auth** — intended for local or trusted use only.
- **PDF quality** — text extraction depends on PDF structure; scanned images need OCR (not included).
- **OpenAI by default** — swap embedding/LLM settings for local models (e.g. Ollama/Llama) in a follow-up.
- **Cost** — indexing and queries use OpenAI API credits.

## Dev container

A `.devcontainer/` config is included for VS Code / GitHub Codespaces development.

## Contributing

Issues and pull requests are welcome on [GitHub](https://github.com/wasimahmadpk/qabot).

## Author

[wasimahmadpk](https://github.com/wasimahmadpk)
