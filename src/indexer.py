import hashlib
from pathlib import Path

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.chunking import chunk_documents, chunk_stats

DEFAULT_CHROMA_PATH = "./chroma_db"


def collection_name_from_signature(upload_signature):
    digest = hashlib.sha256(repr(upload_signature).encode()).hexdigest()
    return f"qabot_{digest[:16]}"


def create_index(
    documents,
    upload_signature,
    chroma_path=DEFAULT_CHROMA_PATH,
    chunk_size=512,
    chunk_overlap=128,
):
    """Chunk documents, embed them, and persist vectors in ChromaDB."""
    nodes = chunk_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    stats = chunk_stats(nodes)

    Path(chroma_path).mkdir(parents=True, exist_ok=True)
    collection_name = collection_name_from_signature(upload_signature)

    chroma_client = chromadb.PersistentClient(path=chroma_path)
    chroma_collection = chroma_client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    if chroma_collection.count() > 0:
        index = VectorStoreIndex.from_vector_store(vector_store)
        return index, stats

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes, storage_context=storage_context)
    return index, stats
