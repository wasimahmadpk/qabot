from pathlib import Path

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.chunking import chunk_documents, chunk_stats
from src.rag_config import collection_name_from_key

DEFAULT_CHROMA_PATH = "./chroma_db"


def create_index(
    documents,
    index_key,
    chroma_path=DEFAULT_CHROMA_PATH,
    chunk_size=512,
    chunk_overlap=128,
    chunk_strategy="sentence",
):
    """Chunk documents, embed them, and persist vectors in ChromaDB."""
    nodes = chunk_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chunk_strategy=chunk_strategy,
    )
    stats = chunk_stats(
        nodes,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chunk_strategy=chunk_strategy,
    )

    Path(chroma_path).mkdir(parents=True, exist_ok=True)
    collection_name = collection_name_from_key(index_key)

    chroma_client = chromadb.PersistentClient(path=chroma_path)
    chroma_collection = chroma_client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    if chroma_collection.count() > 0:
        index = VectorStoreIndex.from_vector_store(vector_store)
        return index, stats

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes, storage_context=storage_context)
    return index, stats
