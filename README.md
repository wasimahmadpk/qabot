# QABot

**QABot** is a small Streamlit app for **retrieval-augmented question answering (RAG)** over your own files. Upload PDF, TXT, or DOCX documents, then ask questions in natural language. Answers are generated with **OpenAI** models and **LlamaIndex** vector search over embedded document chunks.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.45-red)
![LlamaIndex](https://img.shields.io/badge/llamaindex-0.12-green)

## Features

- **Multi-file upload** — index several documents in one session
- **Supported formats** — `.pdf`, `.txt`, `.docx`
- **RAG pipeline** — chunk → embed → retrieve → answer with an LLM
- **Session caching** — re-indexing only runs when uploads change (not on every Streamlit rerun)
- **Simple UI** — upload, ask, read the answer

## How it works

```
Upload files → load & parse text → VectorStoreIndex (LlamaIndex)
                                        ↓
User question → retrieve relevant chunks → OpenAI LLM → answer
```

| Step | Module | Role |
|------|--------|------|
| Load | `src/loader.py` | Parse PDF (PyMuPDF), TXT, DOCX into LlamaIndex `Document` objects |
| Index | `src/indexer.py` | Build an in-memory `VectorStoreIndex` |
| Query | `src/query_engine.py` | Run the index query engine |
| Cache | `src/upload_cache.py` | SHA-256 signature so unchanged uploads skip rebuild |

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

LlamaIndex uses OpenAI for embeddings and the chat model by default. You need a valid [OpenAI API key](https://platform.openai.com/api-keys) with billing enabled.

### 3. Run the app

```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

### 4. Use the app

1. Upload one or more documents (PDF, TXT, or DOCX).
2. Wait for the success message confirming indexing.
3. Type a question about the content and submit.

## Project structure

```
qabot/
├── app.py                 # Streamlit UI
├── assets/logo.png        # App logo
├── src/
│   ├── loader.py          # Document parsing
│   ├── indexer.py         # Vector index creation
│   ├── query_engine.py    # Q&A interface
│   └── upload_cache.py    # Upload change detection
├── tests/
│   └── test_upload_cache.py
├── requirements.txt
└── .env                   # API keys (not committed)
```

## Tests

```bash
python -m unittest discover -s tests -v
```

## Requirements

- Python **3.10+** recommended
- **OpenAI API key** (embeddings + LLM)
- Enough disk/RAM for PyTorch and sentence-transformers (pulled in via LlamaIndex dependencies)

## Limitations

- **In-memory index** — data is not persisted; restarting the app requires re-uploading files.
- **No auth** — intended for local or trusted use only.
- **PDF quality** — text extraction depends on PDF structure; scanned images need OCR (not included).
- **Cost** — each index build and query uses OpenAI API credits.

## Dev container

A `.devcontainer/` config is included for VS Code / GitHub Codespaces development.

## Contributing

Issues and pull requests are welcome on [GitHub](https://github.com/wasimahmadpk/qabot).

## Author

[wasimahmadpk](https://github.com/wasimahmadpk)
