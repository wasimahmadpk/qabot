from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 128


def chunk_documents(
    documents,
    chunk_size=DEFAULT_CHUNK_SIZE,
    chunk_overlap=DEFAULT_CHUNK_OVERLAP,
):
    """Split documents into overlapping chunks for vector retrieval."""
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    nodes = splitter.get_nodes_from_documents(documents)

    per_file_counter = {}
    for node in nodes:
        file_name = node.metadata.get("file_name", "unknown")
        per_file_counter[file_name] = per_file_counter.get(file_name, 0) + 1
        node.metadata["chunk_id"] = per_file_counter[file_name]

    return nodes


def chunk_stats(nodes):
    """Summarize chunk counts for UI and logging."""
    chunks_by_file = {}
    for node in nodes:
        file_name = node.metadata.get("file_name", "unknown")
        chunks_by_file[file_name] = chunks_by_file.get(file_name, 0) + 1

    return {
        "total_chunks": len(nodes),
        "chunks_by_file": chunks_by_file,
    }


def documents_from_nodes(nodes):
    """Rebuild Document list from nodes (useful in tests)."""
    return [
        Document(text=node.get_content(), metadata=dict(node.metadata))
        for node in nodes
    ]
