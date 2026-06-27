# QABot

**QABot** is a Streamlit app for **retrieval-augmented generation (RAG)** over your own documents. Upload PDF, TXT, or DOCX files, ask questions in natural language, and get **grounded answers with source citations**. Vectors persist in **ChromaDB**; embeddings and answers use **OpenAI** via **LlamaIndex**.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.45-red)
![LlamaIndex](https://img.shields.io/badge/llamaindex-0.12-green)

## What it does

| Tab | Purpose |
|-----|---------|
| **Ask Questions** | Upload docs, index into ChromaDB, query with citations |
| **Evaluate RAG** | Run a golden Q&A suite with keyword, IR, and optional RAGAS metrics |

The `src/` pipeline is UI-agnostic — the same ingest, index, and query modules can back a REST API, MCP server, or help portal.

## Features

- **Multi-file upload** — index several documents in one session (`.pdf`, `.txt`, `.docx`)
- **Tunable RAG settings** — chunk strategy, size, overlap, and top-k from the sidebar
- **Sentence-aware chunking** — respects sentence boundaries by default; token-based splitting also available
- **Grounded prompts** — answer only from retrieved context; cite `[file_name, chunk_id]`; say "I don't know" when evidence is missing
- **ChromaDB persistence** — vectors survive app restarts for the same upload + config signature
- **Upload caching** — SHA-256 signature skips re-embedding when files and chunk settings are unchanged
- **RAG evaluation suite** — latency, retrieval hit rate, answer grounding, and IR metrics (Recall@k, MRR, NDCG@k)
- **RAGAS evaluation** — optional LLM-as-judge metrics (faithfulness, answer relevancy, context recall)
- **Custom eval sets** — upload, edit, or reset evaluation JSON without leaving the app

## Architecture

QABot has two distinct phases. **Indexing** runs when you upload files or change chunk settings. **Querying** runs on every new question and does not re-parse, re-chunk, or re-embed your documents.

```
INDEXING (once per upload / chunk config)
Upload → parse → chunk → embed documents → ChromaDB

QUERY (every question)
Question → embed query → retrieve top-k from ChromaDB → grounded prompt → LLM → answer + sources

EVALUATION (on demand)
Golden Q&A → keyword checks + IR metrics + optional RAGAS
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

### Models

QABot uses LlamaIndex and RAGAS defaults — no model names are hard-coded in the app.

| Stage | Model | Provider |
|-------|-------|----------|
| Document embeddings | `text-embedding-ada-002` | OpenAI (LlamaIndex default) |
| Answer generation | `gpt-3.5-turbo` | OpenAI (LlamaIndex default) |
| RAGAS judge (optional) | `gpt-4o-mini` | OpenAI (RAGAS default) |

At query time, only the **user's question** is embedded. Stored document vectors are read from ChromaDB.

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

Each new question reuses the stored index. Documents are not re-embedded unless you upload new files or change chunk settings.

### 5. Run evaluation (optional)

1. Upload `eval/sample_policy.txt` (or your own docs) on the **Ask Questions** tab.
2. Open the **Evaluate RAG** tab.
3. Click **Run evaluation** for keyword and IR metrics, or **Run RAGAS** for LLM-as-judge scoring.

Download buttons on the eval tab provide the sample handbook and default `qa_pairs.json`.

## RAG settings (sidebar)

Engineer controls live in the sidebar under **RAG settings (engineer)**:

| Setting | Default | Notes |
|---------|---------|-------|
| Chunk strategy | `sentence` | `sentence` respects boundaries; `token` uses fixed token windows |
| Chunk size | 512 | Target tokens per chunk; triggers re-indexing when changed |
| Chunk overlap | 128 | Shared tokens between neighboring chunks; triggers re-indexing when changed |
| Top-k | 3 | Chunks retrieved per question; applies immediately without re-indexing |

Changing chunk settings creates a new ChromaDB collection keyed by upload signature + config. Use **Reset RAG defaults** to restore defaults.

## Chunking

Chunking splits each document into smaller, overlapping segments so retrieval can find focused passages instead of whole files.

| Strategy | Splitter | Behavior |
|----------|----------|----------|
| `sentence` (default) | LlamaIndex `SentenceSplitter` | Splits at sentence boundaries up to the chunk size |
| `token` | LlamaIndex `TokenTextSplitter` | Fixed token windows; may cut mid-sentence |

Each chunk receives metadata:

- `file_name` — source document
- `chunk_id` — sequential ID within that file (used in citations)

Short documents that fit within the chunk size stay as a single chunk. Overlap (default 128 tokens) reduces the risk of losing context at chunk boundaries.

## Evaluation

The eval suite runs golden questions from `eval/qa_pairs.json` (or a custom set).

### End-to-end metrics

| Metric | Meaning |
|--------|---------|
| **Retrieval hit rate** | Did retrieved chunks contain all expected keywords? |
| **Grounded answers** | Did the answer include expected facts (or "I don't know" for out-of-scope items)? |
| **Avg latency** | End-to-end query time in milliseconds |

### IR metrics (pure retrieval ranking)

Computed from ranked top-k retrieval **without** running the LLM. All three metrics share the same `top_k` cutoff from sidebar settings.

| Metric | Meaning |
|--------|---------|
| **Recall@k** | Fraction of relevant chunks found in the top-k results |
| **MRR** | Reciprocal rank of the first relevant chunk in top-k (0 if none found) |
| **NDCG@k** | Normalized ranking quality in the top-k list |

When comparing across different retrieval depths, you can write **MRR@k** alongside Recall@k and NDCG@k for clarity. In this app, all three already use the same `top_k`.

### RAGAS metrics (optional)

Click **Run RAGAS** on the Evaluate tab for LLM-as-judge scoring. Requires extra OpenAI API calls.

| Metric | Meaning |
|--------|---------|
| **Faithfulness** | Is the answer supported by retrieved context? |
| **Answer relevancy** | Does the answer address the question? |
| **Context recall** | Does retrieval cover the reference answer? (only when `ground_truth` is set) |

### Eval JSON format

Each eval item supports:

```json
{
  "question": "How many PTO days can carry over to the next year?",
  "expected_keywords": ["5 days"],
  "file_name": "sample_policy.txt",
  "ground_truth": "Up to 5 days of PTO can carry over.",
  "relevant_chunk_ids": [12],
  "refusal": false
}
```

| Field | Purpose |
|-------|---------|
| `expected_keywords` | Facts retrieval and answer checks should contain |
| `file_name` | Optional; restrict checks to a specific source file |
| `ground_truth` | Optional; enables RAGAS context recall |
| `relevant_chunk_ids` | Optional; precise IR metric grading by chunk ID |
| `refusal` | Optional; expects "I don't know" for out-of-scope questions |

Upload custom JSON, edit in the text area, or reset to the default set from the **Eval set & downloads** panel.

## Design decisions

**Chunking (512 / 128)** — balances recall (enough context per chunk) with precision (smaller chunks reduce irrelevant retrieval noise). Overlap avoids cutting sentences at boundaries.

**Grounded prompts** — the LLM must answer only from retrieved context and refuse when evidence is missing. This directly targets hallucination reduction.

**ChromaDB** — persistent vector storage keyed by upload signature and chunk config. Restarting Streamlit does not require re-embedding the same files with the same settings.

**Separate indexing and query paths** — document embeddings are stored once; each question only embeds the query, retrieves from ChromaDB, and calls the LLM.

**Evaluation** — keyword checks and IR metrics are lightweight and deterministic. RAGAS adds LLM-as-judge scoring for deeper quality signals. Together they support regression testing without manual review on every change.

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
- **OpenAI API key** (embeddings + LLM + optional RAGAS judging)
- Disk space for ChromaDB and Python ML dependencies

## Limitations

- **No auth** — intended for local or trusted use only
- **PDF quality** — text extraction depends on PDF structure; scanned images need OCR (not included)
- **OpenAI by default** — swap embedding/LLM settings for local models (e.g. Ollama) in a follow-up
- **Dense retrieval only** — no BM25, hybrid search, or reranking
- **Eval keywords** — simple substring checks; RAGAS helps but neither replaces human grading at scale
- **Cost** — indexing, queries, and RAGAS evaluation use OpenAI API credits

## Dev container

A `.devcontainer/` config is included for VS Code and GitHub Codespaces.

## Contributing

Issues and pull requests are welcome on [GitHub](https://github.com/wasimahmadpk/qabot).

## Author

[wasimahmadpk](https://github.com/wasimahmadpk)
