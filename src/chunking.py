from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter

from src.rag_config import (
    CHUNK_STRATEGIES,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_STRATEGY,
)


def get_splitter(chunk_strategy, chunk_size, chunk_overlap):
    if chunk_strategy == "token":
        return TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    if chunk_strategy != "sentence":
        raise ValueError(f"Unsupported chunk strategy: {chunk_strategy}")
    return SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def chunk_documents(
    documents,
    chunk_size=DEFAULT_CHUNK_SIZE,
    chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    chunk_strategy=DEFAULT_CHUNK_STRATEGY,
):
    """Split documents into overlapping chunks for vector retrieval."""
    splitter = get_splitter(chunk_strategy, chunk_size, chunk_overlap)
    nodes = splitter.get_nodes_from_documents(documents)

    per_file_counter = {}
    for node in nodes:
        file_name = node.metadata.get("file_name", "unknown")
        per_file_counter[file_name] = per_file_counter.get(file_name, 0) + 1
        node.metadata["chunk_id"] = per_file_counter[file_name]

    return nodes


def chunk_stats(nodes, chunk_size=None, chunk_overlap=None, chunk_strategy=None):
    """Summarize chunk counts for UI and logging."""
    chunks_by_file = {}
    for node in nodes:
        file_name = node.metadata.get("file_name", "unknown")
        chunks_by_file[file_name] = chunks_by_file.get(file_name, 0) + 1

    stats = {
        "total_chunks": len(nodes),
        "chunks_by_file": chunks_by_file,
    }
    if chunk_size is not None:
        stats["chunk_size"] = chunk_size
    if chunk_overlap is not None:
        stats["chunk_overlap"] = chunk_overlap
    if chunk_strategy is not None:
        stats["chunk_strategy"] = chunk_strategy
    return stats


def documents_from_nodes(nodes):
    """Rebuild Document list from nodes (useful in tests)."""
    return [
        Document(text=node.get_content(), metadata=dict(node.metadata))
        for node in nodes
    ]
