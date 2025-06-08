# from llama_index import VectorStoreIndex
from llama_index.core import VectorStoreIndex


def create_index(documents):
    return VectorStoreIndex.from_documents(documents)
